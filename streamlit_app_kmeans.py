# Import necessary libraries
import streamlit as st
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import pandas as pd
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import KMeans
import colorcet as cc


############################################
# user def functions
############################################
def load_toy_dataset(n_centers, std, rng, dataset_name='blobs'):
    """
    Load a toy dataset from scikit-learn.

    Parameters:
    - dataset_name (str): Name of the dataset to load ('iris', 'blobs', or 'stars').

    Returns:
    - data (numpy.ndarray): The feature data.
    - target (numpy.ndarray): The target labels.
    """

    if dataset_name == 'iris':
        data = datasets.load_iris().data
        target = datasets.load_iris().target
    elif dataset_name == 'blobs':
        data, target = datasets.make_blobs(n_samples=300, cluster_std=std, centers=n_centers, random_state=rng)
    elif dataset_name == 'moons':
        data, target = datasets.make_moons(n_samples=300, random_state=42)

    return data, target

def plot_dataset(data, target, dataset_name):
    """
    Plot the selected dataset using Matplotlib.

    Parameters:
    - data (numpy.ndarray): The feature data.
    - target (numpy.ndarray): The target labels.
    - dataset_name (str): Name of the dataset.
    """
    
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    
    norm = matplotlib.colors.Normalize(vmin=np.min(target), vmax=np.max(target))
    
    cmap = matplotlib.cm.get_cmap('viridis')

    xmin = np.min(data[:,0])
    xmax = np.max(data[:,0])
    ymin = np.min(data[:,1])
    ymax = np.max(data[:,1])

    ax1.scatter(data[:, 0],data[:, 1], c=target)
    ax1.set_title(f"{dataset_name} Dataset")
    ax1.set_xlabel("Feature 0")
    ax1.set_ylabel("Feature 1")
    for i in np.unique(target):
        ax1.scatter([],[], color=cmap(norm(i)), label=str(i))
    ax1.scatter(centroids[:,0], centroids[:,1],  color='black', marker='^', label='Centroids')
    ax1.legend()
    ax1.set_ylim(ymin, ymax)
    ax1.set_xlim(xmin, xmax)
    st.pyplot(fig1)

def plot_kmeans(data, centroids, step):
    norm = matplotlib.colors.Normalize(vmin=np.min(target), vmax=np.max(target))
    
    cmap = matplotlib.cm.get_cmap('viridis')
    
    xmin = np.min(data[:,0])
    xmax = np.max(data[:,0])
    ymin = np.min(data[:,1])
    ymax = np.max(data[:,1])
    resolution = 40

    inertia = []

    XX, YY = np.meshgrid(np.linspace(xmin, xmax, resolution), np.linspace(ymin, ymax, resolution))

    fig2, ax2 = plt.subplots(figsize=(8, 6))
    ax2.scatter(data[:, 0],data[:, 1], c=target)
    for i in range(step-1):

        model = KMeans(n_clusters=centroids.shape[0], init=centroids, max_iter=i+1)
        model = model.fit(data[:,:2])
        new_centroids = model.cluster_centers_
        ax2.scatter(new_centroids[:,0], new_centroids[:,1] , c = 'red', alpha=0.5, marker = '^')
        inertia.append(model.inertia_)
    model = KMeans(n_clusters=centroids.shape[0], init=centroids, max_iter=step)
    model = model.fit(data[:,:2])
    new_centroids = model.cluster_centers_
    inertia.append(model.inertia_)
    background = model.predict(np.hstack([XX.reshape(-1,1), YY.reshape(-1,1)])).reshape(resolution,resolution)
    ax2.imshow(background, extent=[xmin, xmax, ymin, ymax], cmap=cc.cm.glasbey_cool, origin='lower', aspect='auto', alpha=0.6)
    ax2.scatter(new_centroids[:,0], new_centroids[:,1] , c = 'red', marker = '^')
    
    ax2.scatter(centroids[:,0], centroids[:,1] , c = 'black', marker = '^', label='Orig. pos')
    ax2.set_xlabel("Feature 0")
    ax2.set_ylabel("Feature 1")

    ax2.scatter([], [] , color = 'red', alpha=0.5, marker = '^', label='Prev. pos')
    ax2.scatter([], [] , color = 'red', marker = '^', label='Curr. pos')
    ax2.set_ylim(ymin, ymax)
    ax2.set_xlim(xmin, xmax)
    ax2.legend()
    st.pyplot(fig2)

    return inertia

def plot_inertia(inertia_list):

    fig3, ax3 = plt.subplots(figsize=(8,6))

    ax3.plot(np.arange(1, len(inertia_list)+1), inertia_list, marker='o')
    ax3.set_xlabel('Iteration #')
    ax3.set_ylabel('Inertia')
    ax3.set_title('Inertia per iteration')

    st.pyplot(fig3)         

############################################
# user def functions end
############################################

############################################
# Intro
############################################

st.markdown('This WebApp was developed by CG3 at RWTH Aachen University (Nils Chudalla). It was developed for educational purposes only. Please reference accordingly.')
st.title("KMeans clustering")
st.markdown('Use this webapp to perform K-Means on different classification toy datasets. We use K-Means by sci-kit learn. The first interactive widget lets you to select a reduce the dataset size. The second widget allows you to control the amount and position of clusters fed into the algorithm. The resulting clusters are then plotted. The third widget lets you move through the different steps of the iteration process.')
st.markdown('''
            The K-Means algorithm generally operates as follows:
            ''')
st.markdown('''
            0. Set inital position of centroids (usually with optimization*).
            1. Compute closest points (Inertia).
            2. Move centroid to center of mass of points of corresp. cluster.
            3. Repeat 1. until centroids converge against steady position. 
            ''')
st.markdown('*This example uses user defined centroid positions. The user randomizes the centroid positions')
st.markdown('''
            Unlike agglomerateive clustering, cluster "boundaries" are flexible. Once formed clusters can be changed and it has to be assured that convergence occured. ''')

############################################
# Select dataset
############################################

std = 0
n_centers = 3
rng = 42
# Add a dropdown to select the dataset
st.subheader('Select dataset')
selected_dataset = st.selectbox("Select a dataset", ['blobs', 'iris', 'moons'] )
if 'rng' not in st.session_state:
    st.session_state['rng'] = 42
if selected_dataset == 'blobs':
    col1, col2 = st.columns(2)
    
    n_centers = st.number_input('Enter blob count', value=3, min_value=2, max_value=6)
    with col1:
        if st.button('Randomize blobs'):
            st.session_state['rng'] = st.session_state['rng']+1
    with col2:
        std = st.slider(
            label='Set std.',
            min_value=0.3,
            max_value=2.0,
            value=2.0,
            step=0.2
        )    




# Load the selected dataset
data, target = load_toy_dataset(n_centers, std, st.session_state['rng'], selected_dataset)

df = pd.DataFrame(data, columns=np.arange(data.shape[1]))
df['target'] = target

with st.expander("See data details"):
    st.write(f"Loaded {selected_dataset.upper()} dataset")

    if selected_dataset == 'iris':
        st.markdown('''
                The features correspond with the following observations: 
                - 0 = sepal length in cm
                - 1 = sepal width in cm
                - 2 = petal length in cm
                - 3 = petal width in cm
                ''')
        st.markdown('''
                The values of the target correspond with the following species: 
                - 0 = iris setosa
                - 1 = iris versicolor
                - 2 = iris veriginica
        ''')

    # Display some information about the loaded dataset

    st.write(f"Number of samples: {data.shape[0]}")
    st.write(f"Number of features: {data.shape[1]}")
    st.write(f"Number of classes: {len(set(target))}")


    # Display the loaded data and target
    st.write("Loaded dataset:")
    st.write(df)

############################################
# Refine selection
############################################

selection = np.arange(data.shape[0])
np.random.shuffle(selection)

sel = st.slider(
    label='Reduce selection',
    min_value=0,
    max_value=data.shape[0]-1,
    value=data.shape[0]-1,
    step=1
)

data = data[:sel, :]
target = target[:sel]

############################################
# Refine starting centroids
############################################

k = st.slider(
    label='define amount of clusters',
    min_value=1,
    max_value=8,
    value=2,
    step=1)


xmin = np.min(data[:,0])
xmax = np.max(data[:,0])
ymin = np.min(data[:,1])
ymax = np.max(data[:,1])

seed = 0
x1 = data[:,0]
x2 = data[:,1]

dx1 = (np.max(x1) - np.min(x1))
dx2 = (np.max(x2) - np.min(x2))

centroids_pos = np.array([np.min(x1),np.min(x2)]) + (np.random.random_sample((k, 2))) * np.array([dx1, dx2])

st.subheader('Set centroid positions')
st.markdown('''
With this widget you can define a number of centroids and randomize their initial positions. 

''')
if 'centroids' not in st.session_state:
    st.session_state['centroids'] = centroids_pos

if st.button('Randomize positions'):
    centroids_pos = np.array([np.min(x1),np.min(x2)]) + (np.random.random_sample((k, 2))) * np.array([dx1, dx2])
    st.session_state['centroids'] = centroids_pos 

centroids = st.session_state['centroids']
plot_dataset(data, target, selected_dataset)

############################################
# Iterate algorithm and show inertia
############################################

st.subheader('Iterating the algorithm')
st.markdown('''
This section lets you iterate the kmeans algorithm (max. 20 iterations), based on your starting positions. You can see previous centroid positions as well as the "boundaries" of each cluster behind the scattered points. 
''')
st.markdown('Optimization criterion is:')
st.latex(r'''J=\sum_{i=1}^{k}\sum_{x_{j}\epsilon S_{i}}^{} \left \| x_{j}-\mu_{i}  \right \|''')
st.markdown('Which describes the sum of distances between cluster center and associated points, for all clusters combined. This is called "inertia"')
st.markdown("The bottom figure shows the model's inertia at each step. Note that the y-axis is cut off, making vertical changes more dramatic")

step = st.number_input('Enter iteration number', value=1, min_value=1, max_value=20)

inertia_list = plot_kmeans(data, centroids, step)

plot_inertia(inertia_list)
