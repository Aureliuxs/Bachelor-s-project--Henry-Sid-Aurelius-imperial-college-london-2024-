import optuna
import numpy as np
import os
import time
from functools import partial
import torch
import random
import datetime

# Modules:
from dataset_handling import DatasetHandler
from alb_s1 import ALBERTSentimentFineTuner, ALBERTSentimentInferencer, SentimentCSVDataSaver
from lda_module import LDATextProcessor, LDAProcessor, LDACSVDataSaver
from bertopic_module import BERTopicProcessor, BERTopicCSVDataSaver, TextProcessing
from kcluster_module import TextProcessingAndDatabase, KclusterAnalysis, KclusterCSVDataSaver

# Visualization
import matplotlib.pyplot as plt
import optuna.visualization as ov

# Parallelization imports
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

# Set up logging
import logging
logging.basicConfig(level=logging.INFO)


### Albert Sentiment Analysis

def optimize_albert_hyperparameters(dataset_name='stocksentiment', n_trials=3, n_jobs=1):
    """
    Optimize hyperparameters for ALBERT model fine-tuning using Optuna.

    Args:
        dataset_name (str): Name of the dataset to use.
        n_trials (int): Number of optimization trials.
        n_jobs (int): Number of parallel jobs.

    Returns:
        dict: Best hyperparameters found.
    """
    import time
    start_time = time.time()  # Time process

    # Random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    fine_tuner = ALBERTSentimentFineTuner(max_seq_length=128, model_size='base')

    # Prepare dataset
    dataset = {
        'train': fine_tuner.prepare_finetuning_dataset(
            dataset_name=dataset_name,
            split='train',
            sample_size=1000,
            shuffle_data=True
        ),
        'test': fine_tuner.prepare_finetuning_dataset(
            dataset_name=dataset_name,
            split='test',
            sample_size=200,
            shuffle_data=True
        )
    }

    def compute_metrics(pred):
        from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        precision = precision_score(labels, preds, average='weighted', zero_division=0)
        recall = recall_score(labels, preds, average='weighted', zero_division=0)
        f1 = f1_score(labels, preds, average='weighted', zero_division=0)
        acc = accuracy_score(labels, preds)

        return {
            'accuracy': acc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
        }

    def objective(trial):
        # Suggest hyperparameters
        learning_rate = trial.suggest_loguniform('learning_rate', 1e-6, 1e-4)
        batch_size = trial.suggest_categorical('batch_size', [8, 16, 32])
        epochs = trial.suggest_int('epochs', 1, 5)
        weight_decay = trial.suggest_loguniform('weight_decay', 1e-4, 1e-1)

        # Initialize fine-tuner
        fine_tuner.initialize_model()

        # Set hyperparameters
        training_args = fine_tuner.get_training_args(
            learning_rate=learning_rate,
            batch_size=batch_size,
            epochs=epochs,
            weight_decay=weight_decay
        )

        # Initialize Trainer
        trainer = fine_tuner.get_trainer(
            training_args=training_args,
            train_dataset=dataset['train'],
            eval_dataset=dataset['test'],
            compute_metrics=compute_metrics  # Include compute_metrics function
        )

        # Fine-tune the model
        trainer.train()

        # Evaluate the model
        metrics = trainer.evaluate()

        # Debug: Print the metrics
        print(f"Metrics returned from evaluation: {metrics}")

        # Save metrics to a trial
        trial.set_user_attr('accuracy', metrics.get('eval_accuracy'))
        trial.set_user_attr('precision', metrics.get('eval_precision'))
        trial.set_user_attr('recall', metrics.get('eval_recall'))
        trial.set_user_attr('f1', metrics.get('eval_f1'))

        # Return negative accuracy as Optuna minimizes the objective
        return -metrics.get('eval_accuracy')

    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner()
    )

    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)

    # End timer
    end_time = time.time()
    total_time = end_time - start_time

    # Get best trial
    best_trial = study.best_trial

    # Visualization
    ov.plot_optimization_history(study).show()
    ov.plot_param_importances(study).show()

    print("Best hyperparameters:", study.best_params)
    print("Best accuracy score:", -best_trial.value)

    # Collect results
    results = {
        'best_hyperparameters': best_trial.params,
        'best_accuracy': -best_trial.value,
        'precision': best_trial.user_attrs.get('precision'),
        'recall': best_trial.user_attrs.get('recall'),
        'f1': best_trial.user_attrs.get('f1'),
        'total_time_seconds': total_time
    }

    # Collect date for file reference
    current_time = datetime.datetime.now()

    # Save results to a text file
    with open('optimization_results_.txt', 'w') as f:
        f.write("Hyperparameter Optimization Results\n")
        f.write("=" * 40 + "\n")
        f.write(f"Time Performed: {current_time}\n")
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Number of Trials: {n_trials}\n")
        f.write(f"Total Time Taken (seconds): {total_time:.2f}\n")
        f.write("\nBest Hyperparameters:\n")
        for key, value in best_trial.params.items():
            f.write(f"  {key}: {value}\n")
        f.write(f"\nPerformance Metrics on Test Data:\n")
        f.write(f"  Accuracy: {results['best_accuracy']:.4f}\n")
        f.write(f"  Precision: {results['precision']:.4f}\n")
        f.write(f"  Recall: {results['recall']:.4f}\n")
        f.write(f"  F1 Score: {results['f1']:.4f}\n")

    print("Best hyperparameters:", best_trial.params)
    print("Best accuracy score:", -best_trial.value)

    return best_trial.params


### BERTopic Optimization

def optimize_bertopic_hyperparameters(texts, n_trials):
    import optuna
    from gensim.models.coherencemodel import CoherenceModel
    from gensim.corpora.dictionary import Dictionary

    def objective(trial):
        n_neighbors = trial.suggest_int('n_neighbors', 2, 15)
        min_cluster_size = trial.suggest_int('min_cluster_size', 10, 100)

        bertopic_processor = BERTopicProcessor(num_topics=None)
        bertopic_processor.initialize_model(len(texts), n_neighbors=n_neighbors, min_cluster_size=min_cluster_size)
        topics = bertopic_processor.perform_bertopic(texts)

        # Extract the topics and their representations
        topic_info = bertopic_processor.topic_model.get_topic_info()
        topic_list = bertopic_processor.topic_model.get_topics()

        # Prepare data for CoherenceModel
        # Tokenize the texts
        tokenized_texts = [text.split() for text in texts]

        # Create a dictionary and corpus required for CoherenceModel
        dictionary = Dictionary(tokenized_texts)
        corpus = [dictionary.doc2bow(text) for text in tokenized_texts]

        # Extract the words for each topic
        topics_words = []
        for topic_id in topic_info['Topic']:
            if topic_id == -1:
                continue  # Skip outlier topic
            words_probs = topic_list[topic_id]
            words = [word for word, _ in words_probs]
            topics_words.append(words)

        # Calculate the coherence score
        coherence_model = CoherenceModel(
            topics=topics_words,
            texts=tokenized_texts,
            corpus=corpus,
            dictionary=dictionary,
            coherence='c_v'
        )
        coherence_score = coherence_model.get_coherence()

        return coherence_score

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials)

    return study.best_params


### KMeans Clustering Optimization

def optimize_kmeans_hyperparameters(texts, n_trials=20, n_jobs=1):
    """
    Optimize hyperparameters for KMeans clustering using Optuna.

    Args:
        texts (list): List of preprocessed texts.
        n_trials (int): Number of optimization trials.
        n_jobs (int): Number of parallel jobs.

    Returns:
        dict: Best hyperparameters found.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    def objective(trial):
        # Suggest hyperparameters
        max_df = trial.suggest_uniform('max_df', 0.5, 1.0)
        min_df = trial.suggest_uniform('min_df', 0.0, 0.1)
        ngram_range = trial.suggest_categorical('ngram_range', [(1, 1), (1, 2), (1, 3)])
        n_clusters = trial.suggest_int('n_clusters', 2, 50)

        # Vectorize texts
        tfidf_vectorizer = TfidfVectorizer(
            stop_words='english',
            max_df=max_df,
            min_df=min_df,
            ngram_range=ngram_range
        )
        tfidf_matrix = tfidf_vectorizer.fit_transform(texts)

        # Apply KMeans clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(tfidf_matrix)

        # Compute silhouette score
        if len(set(labels)) > 1:
            score = silhouette_score(tfidf_matrix, labels)
            return -score  # Optuna minimizes the objective
        else:
            return float('inf')  # Penalize if only one cluster is found

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)

    # Visualization
    ov.plot_optimization_history(study).show()
    ov.plot_param_importances(study).show()

    print("Best hyperparameters:", study.best_params)
    print("Best silhouette score:", -study.best_value)

    return study.best_params


### LDA Topic Optimization

def optimize_lda_hyperparameters(texts, n_trials=20, n_jobs=1):
    """
    Optimize hyperparameters for LDA using Optuna.

    Args:
        texts (list): List of preprocessed and tokenized texts.
        n_trials (int): Number of optimization trials.
        n_jobs (int): Number of parallel jobs.

    Returns:
        dict: Best hyperparameters found.
    """
    from gensim import corpora
    from gensim.models import CoherenceModel

    # Prepare dictionary and corpus
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]

    def objective(trial):
        # Suggest hyperparameters
        num_topics = trial.suggest_int('num_topics', 2, 50)
        alpha = trial.suggest_categorical('alpha', ['symmetric', 'asymmetric', 'auto'])
        eta = trial.suggest_categorical('eta', ['symmetric', 'auto'])

        # Initialize LDAProcessor
        lda_processor = LDAProcessor(num_topics=num_topics, alpha=alpha, eta=eta)

        # Perform LDA
        lda_model, _, _ = lda_processor.perform_lda(texts)

        # Compute coherence score
        coherence_model = CoherenceModel(
            model=lda_model,
            texts=texts,
            dictionary=dictionary,
            coherence='c_v'
        )
        coherence_score = coherence_model.get_coherence()

        return -coherence_score  # Optuna minimizes the objective

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)

    # Visualization
    ov.plot_optimization_history(study).show()
    ov.plot_param_importances(study).show()

    print("Best hyperparameters:", study.best_params)
    print("Best coherence score:", -study.best_value)

    return study.best_params
