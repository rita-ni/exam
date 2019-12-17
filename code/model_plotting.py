import numpy as np
import pandas as pd
import altair as alt
alt.data_transformers.disable_max_rows()
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB

def plot_tree(X, y, model, predict_proba = False):
    
    # Join data for plotting
    sample = (X.join(y))
    # Create a mesh for plotting
    step = (X.max() - X.min()) / 50
    x1, x2 = np.meshgrid(np.arange(sample.min()[0]-step[0], sample.max()[0]+step[0], step[0]),
                         np.arange(sample.min()[1]-step[1], sample.max()[1]+step[1], step[1]))

    # Store mesh in dataframe
    mesh_df = pd.DataFrame(np.c_[x1.ravel(), x2.ravel()], columns=['x1', 'x2'])

    # Mesh predictions
    if predict_proba:
        mesh_df['predictions'] = model.predict_proba(mesh_df[['x1', 'x2']])[:, 0]
        # Plot
        base_plot = alt.Chart(mesh_df).mark_rect(opacity=0.5).encode(
            x=alt.X('x1', bin=alt.Bin(step=step[0])),
            y=alt.Y('x2', bin=alt.Bin(step=step[1])),
            color=alt.Color('predictions', title='P(blue)', scale=alt.Scale(scheme='redblue'))
        ).properties(
            width=400,
            height=400
        )
        return alt.layer(base_plot).configure_axis(
            labelFontSize=20,
            titleFontSize=20
        ).configure_legend(
            titleFontSize=20,
            labelFontSize=20
        )
    else:
        mesh_df['predictions'] = model.predict(mesh_df[['x1', 'x2']])
        # Plot
        scat_plot = alt.Chart(sample).mark_circle(
            stroke='black',
            opacity=1,
            strokeWidth=1.5,
            size=100
        ).encode(
            x=alt.X(X.columns[0], axis=alt.Axis(labels=True, ticks=True, title=X.columns[0])),
            y=alt.Y(X.columns[1], axis=alt.Axis(labels=True, ticks=True, title=X.columns[1])),
            color=alt.Color(y.columns[0])
        )
        base_plot = alt.Chart(mesh_df).mark_rect(opacity=0.5).encode(
            x=alt.X('x1', bin=alt.Bin(step=step[0])),
            y=alt.Y('x2', bin=alt.Bin(step=step[1])),
            color=alt.Color('predictions', title='Legend')
        ).properties(
            width=400,
            height=400
        )
        return alt.layer(base_plot, scat_plot).configure_axis(
            labelFontSize=20,
            titleFontSize=20
        ).configure_legend(
            titleFontSize=20,
            labelFontSize=20
        )

def plot_knn_regressor(X, y, model):
    
    x_grid = np.linspace(min(X), max(X), 1000)
    y_predicted = np.squeeze(model.predict(x_grid))
    
    df1 = pd.DataFrame({'X': np.squeeze(X),
                        'y': np.squeeze(y)})
    df2 = pd.DataFrame({'X': np.squeeze(x_grid),
                        'y': np.squeeze(y_predicted)})
    
    scatter = alt.Chart(df1
    ).mark_circle(size=100,
                  color='red',
                  opacity=1
    ).encode(
        x='X',
        y='y')

    line = alt.Chart(df2
    ).mark_line(
    ).encode(
        x='X',
        y='y'
    )

    return alt.layer(scatter, line).configure_axis(
            labelFontSize=20,
            titleFontSize=20
        ).configure_legend(
            titleFontSize=20,
            labelFontSize=20
        )

def plot_lowess(X, y, z):
    
    df1 = pd.DataFrame({'X':X,
                        'y':y})
    df2 = pd.DataFrame({'X':z[:,0],
                        'y':z[:,1]})
    scatter = alt.Chart(df1
    ).mark_circle(size=100,
                  color='red',
                  opacity=1
    ).encode(
        x='X',
        y='y')

    line = alt.Chart(df2
    ).mark_line(
    ).encode(
        x='X',
        y='y'
    )

    return alt.layer(scatter, line).configure_axis(
            labelFontSize=20,
            titleFontSize=20
        ).configure_legend(
            titleFontSize=20,
            labelFontSize=20
        )

def plot_laplace_smoothing(X_train, y_train, X_test, y_test, alpha_vals=np.logspace(-10, 1, num=12)):
    train_err = []
    test_err = []
    for alpha in alpha_vals:
        nb = MultinomialNB(alpha=alpha)
        nb.fit(X_train, y_train)
        train_err.append(1-nb.score(X_train, y_train))
        test_err.append(1-nb.score(X_test, y_test))
    
    df = pd.DataFrame({'alpha': alpha_vals,
                       'train_error': train_err,
                       'test_error': test_err}).melt(id_vars='alpha',
                                                     var_name='dataset',
                                                     value_name='error')
    lines = alt.Chart(df
    ).mark_line(
    ).encode(
        x=alt.X('alpha', axis=alt.Axis(format='e'), title='laplace smoothing (alpha)', scale=alt.Scale(type='log')),
        y=alt.Y('error', title='error rate'),
        color='dataset'
    )

    return alt.layer(lines).configure_axis(
            labelFontSize=20,
            titleFontSize=20
        ).configure_legend(
            titleFontSize=20,
            labelFontSize=20
        ).properties(
        width=400,
        height=300
    )