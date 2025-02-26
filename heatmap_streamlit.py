import streamlit as st
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
import io

def parse_pasted_data(pasted_text):
    """
    Parse tab-separated data from clipboard or direct paste
    Expects two rows: first row is column names, second row is values
    Handles apostrophes in column names
    """
    # Use StringIO to simulate a file-like object
    data_stringio = io.StringIO(pasted_text)
    
    # Read the data
    try:
        # Read raw data
        df = pd.read_csv(data_stringio, sep='\t', header=None)
        
        # If two rows, use first as columns, second as values
        if df.shape[0] == 2:
            # Clean column names by stripping apostrophes
            columns = [col.strip("'") for col in df.iloc[0].tolist()]
            values = df.iloc[1].tolist()
            
            # Ensure values are converted to numeric
            values = [float(val) for val in values]
            
            # Create a DataFrame with one row
            parsed_df = pd.DataFrame([values], columns=columns)
            return parsed_df
        else:
            st.error("Please paste exactly two rows: column names and corresponding values")
            return None
    except ValueError:
        st.error("Error: Ensure all values are numeric")
        return None
    except Exception as e:
        st.error(f"Error parsing data: {e}")
        return None

def create_rule66_heatmap(op):
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

    # First, add values from the DataFrame row
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
        return None

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

    return fig

def main():
    st.title("Rule 66 Heatmap Generator")
    
    # Instructions
    st.markdown("""
    ### How to Use
    1. Copy the entire row of column names from Excel (INT11 thru REAL39)
    2. Copy the entire row of corresponding values from Excel
    3. Paste the rows in the text area below (tab-separated)
    
    **Example:**
    ```
    Int11	Int12	Int13	Int14	Int15	Int16	Int17	Int18	Int19	Int20	Int21	Int22	Int23	Int24	Int25	Int26	Int27	Int28	Int29	Int30	Int31	Int32	Int33	Int34	Int35	Int36	Int37	Int38	Int39	Int40	Real1	Real2	Real3	Real4	Real5	Real6	Real7	Real8	Real9	Real10	Real11	Real12	Real13	Real14	Real15	Real16	Real17	Real18	Real19	Real20	Real21	Real22	Real23	Real24	Real25	Real26	Real27	Real28	Real29	Real30	Real31	Real32	Real33	Real34	Real35	Real36	Real37	Real38	Real39
    1000	35000	50000	100000	150000	257000	1000	33000	65000	100000	190000	196000	97	97	95	91	87	84	1	97	97	95	93	91	89	1	953500	553600	1	-1	0	0	0	0	0	0	0	0	0	0	0	97	49	2	1	1	1	1	97	90	50	40	32	19	1	97	95	75	45	39	33	1	97	97	95	66	39	33	1
    ```
    """)
    
    # Text area for data input
    pasted_data = st.text_area("Paste your data here (tab-separated):")
    
    if st.button("Generate Heatmap") and pasted_data:
        # Parse the pasted data
        parsed_df = parse_pasted_data(pasted_data)
        
        if parsed_df is not None:
            # Create the heatmap
            fig = create_rule66_heatmap(parsed_df)
            
            if fig is not None:
                # Display the plot
                st.pyplot(fig)
                
                # Save option
                save_path = st.text_input("Save path (optional):", value="Rule66_Heatmap.png")
                if st.button("Save Heatmap"):
                    plt.savefig(save_path)
                    st.success(f"Heatmap saved to {save_path}")

if __name__ == "__main__":
    main()
