import pandas as pd
import os

def main():
    # Initialize a list to store the dataframes
    df_list = []

    # Go through folders titled result_clipx where x varies from 1 to 20
    for i in range(1, 21):
        folder_name = f'result_clip{i}'

        # Assume there's only one csv file in the folder
        for filename in os.listdir(folder_name):
            if filename.endswith(".csv"):
                csv_file = os.path.join(folder_name, filename)
                
                # Read the CSV file
                df = pd.read_csv(csv_file)
                
                # Filter the dataframe to only keep rows where 'image_name' column equals 'clipx/frame0.png'
                df_filtered = df[df['image_name'] == f'clip{i}/frame0.png']
                
                # Append the filtered dataframe to the list
                df_list.append(df_filtered)

    # Concatenate all the dataframes in the list into one dataframe
    df_all = pd.concat(df_list)

    # Write the final dataframe into a new CSV file
    df_all.to_csv('first_frames_bbx.csv', index=False)

if __name__ == '__main__':
    main()
