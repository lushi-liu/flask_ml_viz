from flask import Flask, render_template
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.io as pio

app = Flask(__name__)

X, _ = make_blobs(n_samples=300, centers=4, random_state=42)
kmeans = KMeans(n_clusters=4, random_state=42)
labels = kmeans.fit_predict(X)
df = pd.DataFrame(X, columns=['Feature 1', 'Feature 2'])
df['Cluster'] = labels

fig = px.scatter(df, x='Feature 1', y='Feature 2', color='Cluster', title='K-Means Clustering')
plot_html = pio.to_html(fig, full_html=False)

@app.route('/')
def index():
    return render_template('index.html', plot=plot_html)

if __name__ == '__main__':
    app.run(debug=True)