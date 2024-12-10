import streamlit as st
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
import os

# Path to the uploaded CSV (Streamlit file uploader will handle this)
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    # Read the CSV file with specific parameters
    op = pd.read_csv(uploaded_file, header=1, encoding='utf-8-sig')

    # Define the fixed values (these are values not in the CSV)
    fixed_values = {
        'Int21': 190000,
        'Int22': 196000,
        'Real11': 0,
        'Real18': 1,
        'Real25': 1,
        'Real32': 1,
        'Real39': 1,
        'Int29': 1,
        'Int36': 1
    }

    # Create the rule66_dict
    rule66_dict = {}

    # First, add values from the CSV row
    for col in op.columns:
        clean_col = col.strip("'")
        rule66_dict[clean_col] = op[col].values[0]

    # Then, update with fixed values
    rule66_dict.update(fixed_values)

    # Verify the dictionary contains all required keys
    required_columns = ['Int16', 'Int22', 'Int11', 'Int12', 'Int13', 'Int14', 'Int15',
                        'Int17', 'Int18', 'Int19', 'Int20', 'Int21']

    missing_columns = [col for col in required_columns if col not in rule66_dict]
    if missing_columns:
        st.error(f"Missing required columns: {missing_columns}")
        st.stop()

    # x and y axis values from the dictionary
    xmin = 0
    xmax = rule66_dict['Int16']
    ymin = 0
    ymax = rule66_dict['Int22']

    # x and y values
    xvals = np.array([0, rule66_dict['Int11'], rule66_dict['Int12'], rule66_dict['Int13'],
                      rule66_dict['Int14'], rule66_dict['Int15'], rule66_dict['Int16']])
    yvals = np.array([0, -rule66_dict['Int17'], -rule66_dict['Int18'], -rule66_dict['Int19'],
                      -rule66_dict['Int20'], -rule66_dict['Int21'], -rule66_dict['Int22']])

    # Table values
    newz2 = np.array([
        [rule66_dict['Real5'], rule66_dict['Real12'], rule66_dict['Real19'], rule66_dict['Real26'],
         rule66_dict['Real33'], rule66_dict['Int23'], rule66_dict['Int30']],
        [rule66_dict['Real6'], rule66_dict['Real13'], rule66_dict['Real20'], rule66_dict['Real27'],
         rule66_dict['Real34'], rule66_dict['Int24'], rule66_dict['Int31']],
        [rule66_dict['Real7'], rule66_dict['Real14'], rule66_dict['Real21'], rule66_dict['Real28'],
         rule66_dict['Real35'], rule66_dict['Int25'], rule66_dict['Int32']],
        [rule66_dict['Real8'], rule66_dict['Real15'], rule66_dict['Real22'], rule66_dict['Real29'],
         rule66_dict['Real36'], rule66_dict['Int26'], rule66_dict['Int33']],
        [rule66_dict['Real9'], rule66_dict['Real16'], rule66_dict['Real23'], rule66_dict['Real30'],
         rule66_dict['Real37'], rule66_dict['Int27'], rule66_dict['Int34']],
        [rule66_dict['Real10'], rule66_dict['Real17'], rule66_dict['Real24'], rule66_dict['Real31'],
         rule66_dict['Real38'], rule66_dict['Int28'], rule66_dict['Int35']],
        [rule66_dict['Real11'], rule66_dict['Real18'], rule66_dict['Real25'], rule66_dict['Real32'],
         rule66_dict['Real39'], rule66_dict['Int29'], rule66_dict['Int36']]
    ])

    # Interpolation
    interpolator = RegularGridInterpolator((xvals, yvals), newz2)

    xnew = np.linspace(xvals.min(), xvals.max(), 100)
    ynew = np.linspace(yvals.min(), yvals.max(), 100)
    Xnew, Ynew = np.meshgrid(xnew, ynew)

    points = np.vstack((Xnew.flatten(), Ynew.flatten())).T
    znew = interpolator(points)
    znew = znew.reshape(Xnew.shape)

    # Plotting the heatmap
    fig, ax = plt.subplots(figsize=(abs((xmax - xmin)) * .00005, abs((ymax - ymin)) * .00005))

    # Plot the heatmap with axis scaling
    c = ax.pcolormesh(Xnew, Ynew, znew, cmap="rainbow_r")
    ax.set_xlim(xvals.min(), xvals.max())  # Ensure the x axis is properly set
    ax.set_ylim(yvals.min(), yvals.max())  # Ensure the y axis is properly set

    # Adjust ticks and labels to match the original visualization
    ax.set_xticks(xvals)
    ax.set_xticklabels([f'{int(x)}' for x in xvals], rotation=30)
    ax.set_yticks(yvals)
    ax.set_yticklabels([f'{int(y)}' for y in yvals], rotation=30)

    # Title and labels
    ax.set_xlabel("Storage in Dillon (af)")
    ax.set_ylabel("E/S Storage (af)")
    ax.set_title("From: Rule66_Heatmap")

    # Add color bar
    fig.colorbar(c, ax=ax, label="% of Total Release from 1st Acct")

    # Show plot in Streamlit
    st.pyplot(fig)
