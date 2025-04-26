import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import multiprocessing
from matplotlib.lines import Line2D
from tqdm import tqdm
import time
import numpy as np

# Set the non-interactive backend
plt.switch_backend('Agg')
os.makedirs("output", exist_ok=True)

# Progress tracking
progress_queue = multiprocessing.Queue()
total_plots = 8  # Total number of plots (4 original + 4 new)

def update_progress():
    """Display progress in the main thread"""
    completed = 0
    with tqdm(total=total_plots, desc="Generating plots") as pbar:
        while completed < total_plots:
            if not progress_queue.empty():
                progress_queue.get()
                completed += 1
                pbar.update(1)
            time.sleep(0.1)

def load_data():
    data = pd.read_csv("data/movie.csv")
    data['Rotten Tomatoes'] = data['Rotten Tomatoes'].str.replace('%', '').astype(float)
    data['IMDb_%'] = data['IMDb'] * 10  # Normalized to 0-100 scale
    return data

# ================= ORIGINAL PLOTS =================
def create_scatter_plot(data, queue):
    try:
        plt.figure(figsize=(12, 8))
        unique_genres = set()
        for genres in data['Genres']:
            unique_genres.update(g.strip() for g in genres.split(','))
        palette = sns.color_palette("husl", len(unique_genres))
        genre_colors = dict(zip(unique_genres, palette))

        for idx, row in data.iterrows():
            genres = [g.strip() for g in row['Genres'].split(',')]
            for genre in genres:
                plt.scatter(
                    x=row['IMDb'],
                    y=row['Rotten Tomatoes'],
                    color=genre_colors[genre],
                    alpha=0.7,
                    s=100
                )

        legend_elements = [Line2D([0], [0], marker='o', color='w', 
                          label=genre, markerfacecolor=color, markersize=10)
                         for genre, color in genre_colors.items()]
        plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.title("IMDb vs. Rotten Tomatoes Ratings")
        plt.xlabel("IMDb Rating")
        plt.ylabel("Rotten Tomatoes Score")
        plt.tight_layout()
        plt.savefig("output/imdb_vs_rotten.png", bbox_inches='tight')
        plt.close()
        queue.put(1)
    except Exception as e:
        print(f"Error in scatter plot: {e}")

def create_runtime_plot(data, queue):
    try:
        plt.figure(figsize=(10, 6))
        sns.histplot(data['Runtime'], bins=10, kde=True, color='skyblue')
        plt.title("Runtime Distribution")
        plt.savefig("output/runtime_distribution.png")
        plt.close()
        queue.put(1)
    except Exception as e:
        print(f"Error in runtime plot: {e}")

def create_genre_plot(data, queue):
    try:
        plt.figure(figsize=(12, 6))
        genres = data['Genres'].str.split(',', expand=True).stack().str.strip()
        genre_counts = genres.value_counts()
        genre_counts.plot(kind='bar', color='orange')
        plt.title("Genre Popularity")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig("output/genre_popularity.png")
        plt.close()
        queue.put(1)
    except Exception as e:
        print(f"Error in genre plot: {e}")

def create_streaming_plot(data, queue):
    try:
        plt.figure(figsize=(8, 5))
        streaming_data = data[['Netflix', 'Hulu', 'Prime Video', 'Disney+']].sum()
        streaming_data.plot(kind='bar', color=sns.color_palette("muted"))
        plt.title("Streaming Availability")
        plt.savefig("output/streaming_platforms.png")
        plt.close()
        queue.put(1)
    except Exception as e:
        print(f"Error in streaming plot: {e}")

# ================= NEW RATINGS PLOTS =================
def create_genre_ratings_heatmap(data, queue):
    try:
        plt.figure(figsize=(12, 8))
        exploded = data.assign(Genres=data['Genres'].str.split(',')).explode('Genres')
        exploded['Genres'] = exploded['Genres'].str.strip()
        genre_ratings = exploded.groupby('Genres')[['IMDb_%', 'Rotten Tomatoes']].mean()
        
        sns.heatmap(genre_ratings, annot=True, cmap='YlOrRd', fmt='.1f')
        plt.title("Average Ratings by Genre")
        plt.ylabel("Genre")
        plt.xlabel("Rating System")
        plt.tight_layout()
        plt.savefig("output/genre_ratings_heatmap.png", bbox_inches='tight')
        plt.close()
        queue.put(1)
    except Exception as e:
        print(f"Error in genre ratings heatmap: {e}")

def create_rating_distribution(data, queue):
    try:
        plt.figure(figsize=(10, 6))
        ratings = data.melt(id_vars=['Title'], 
                          value_vars=['IMDb_%', 'Rotten Tomatoes'],
                          var_name='Rating System', 
                          value_name='Score')
        
        sns.boxplot(x='Rating System', y='Score', data=ratings, palette='Set2')
        plt.title("Rating Distributions")
        plt.xlabel("Rating System")
        plt.ylabel("Score (%)")
        plt.ylim(0, 100)
        plt.savefig("output/rating_distributions.png", bbox_inches='tight')
        plt.close()
        queue.put(1)
    except Exception as e:
        print(f"Error in rating distribution plot: {e}")

def create_top_rated_movies(data, queue):
    try:
        plt.figure(figsize=(12, 6))
        data['Combined_Rating'] = (data['IMDb_%'] + data['Rotten Tomatoes']) / 2
        top_movies = data.nlargest(10, 'Combined_Rating')[['Title', 'Genres', 'Combined_Rating']]
        
        sns.barplot(x='Combined_Rating', y='Title', data=top_movies, palette='viridis')
        plt.title("Top Rated Movies by Combined Score")
        plt.xlabel("Combined Rating Score (%)")
        plt.ylabel("Movie Title")
        
        for i, (_, row) in enumerate(top_movies.iterrows()):
            plt.text(5, i, row['Genres'].split(',')[0], ha='left', va='center')
        
        plt.tight_layout()
        plt.savefig("output/top_rated_movies.png", bbox_inches='tight')
        plt.close()
        queue.put(1)
    except Exception as e:
        print(f"Error in top rated movies plot: {e}")

def create_director_ratings(data, queue):
    try:
        plt.figure(figsize=(12, 6))
        director_ratings = data.groupby('Directors')['IMDb'].mean().nlargest(5)
        director_ratings.plot(kind='bar', color='purple')
        plt.title("Top Directors by IMDb Rating")
        plt.savefig("output/top_directors.png")
        plt.close()
        queue.put(1)
    except Exception as e:
        print(f"Error in director ratings plot: {e}")

# ================= MAIN EXECUTION =================
if __name__ == '__main__':
    data = load_data()
    
    # Start progress updater
    progress_thread = multiprocessing.Process(target=update_progress)
    progress_thread.start()
    
    # Create processes for all plots
    processes = []
    plot_functions = [
        # Original plots
        create_scatter_plot,
        create_runtime_plot,
        create_genre_plot,
        create_streaming_plot,
        # New ratings plots
        create_genre_ratings_heatmap,
        create_rating_distribution,
        create_top_rated_movies,
        create_director_ratings
    ]
    
    # Start all plot processes
    for func in plot_functions:
        p = multiprocessing.Process(target=func, args=(data, progress_queue))
        processes.append(p)
        p.start()
    
    # Wait for completion
    for p in processes:
        p.join()
    
    progress_thread.join()
    
    print("\nAll visualizations generated successfully!")
    print("Original plots:")
    print("- imdb_vs_rotten.png, runtime_distribution.png")
    print("- genre_popularity.png, streaming_platforms.png")
    print("\nNew ratings analysis:")
    print("- genre_ratings_heatmap.png, rating_distributions.png")
    print("- top_rated_movies.png, top_directors.png")