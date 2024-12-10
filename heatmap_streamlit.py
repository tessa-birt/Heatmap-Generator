import streamlit as st
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt

# Create a text area for user to paste the table data
pasted_data = st.text_area("Paste the table data here (Tab-delimited format)", height=300)

if pasted_data:
    # Convert the pasted data to a pandas DataFrame
    try:
        # Split the pasted data by new lines and tabs to mimic how the table is structured
        data_lines = pasted_data.strip().split('\n')
        data_list = [line.split('\t') for line in data_lines]

        # Convert the list of lists into a DataFrame
        df = pd.DataFrame(data_list[1:], columns=data_list[0])

        # Strip any leading/trailing spaces from column names
        df.columns = df.columns.str.strip()

        # Ensure the columns are numeric for calculations (convert them to float)
        df = df.apply(pd.to_numeric, errors='coerce')

        # Display the DataFrame (for debugging)
        st.write(df)

        # Extract necessary values (e.g., Real5, Real6, etc.)
        rule66_dict = {col: df[col].iloc[0] for col in df.columns if 'Real' in col or 'Int' in col}

        # Define fixed values for Int21 and Int22
        fixed_values = {
            'Int21': 190000,
            'Int22': 196000
        }

        # Update the dictionary with fixed values
        rule66_dict.update(fixed_values)

        # Verify the dictionary contains all required keys (excluding Int21 and Int22)
        required_columns = ['Int16', 'Int11', 'Int12', 'Int13', 'Int14', 'Int15', 
                            'Int17', 'Int18', 'Int19', 'Int20']

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

        # Table values (convert each Real column to a row)
        newz2 = np.array([rule66_dict[col] for col in df.columns if 'Real' in col]).reshape(7, 7)

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

        # Optional: Save the heatmap if desired
        save_button = st.button("Save Heatmap")
        if save_button:
            save_path = st.text_input("Enter the save path:", value="Rule66_Heatmap.png")
            plt.savefig(save_path)
            st.success(f"Heatmap saved to {save_path}")

    except Exception as e:
        st.error(f"An error occurred while processing the data: {e}")
