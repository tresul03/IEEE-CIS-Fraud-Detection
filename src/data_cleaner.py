import pandas as pd
import logging
from logging_config import setup_logging


class DataCleaner:

    def __init__(
        self,
        target_col: str,
    ):
        self.target_col = target_col

        # Initialize the logger
        self.logger = setup_logging(__name__)

    def clean_dataset(self, df: pd.DataFrame):
        """
        Clean the dataset by removing missing values and balancing the dataset.

        Process:
        1. Remove missing values from the DataFrame.
        2. Implement balanced sampling on the DataFrame.
        3. Log the number of samples for each class and return the balanced
        DataFrame.

        Parameters:
        ``df``: DataFrame to clean

        Returns:
        ``df``: Cleaned DataFrame
        """

        self.logger.info("Cleaning the dataset")
        # df = self.balanced_sampling(df)
        df = self.remove_missing_values(df)

        class_counts = df[self.target_col].value_counts()
        self.logger.info(
            f"Number of samples for each class: {class_counts[0]} and \
            {class_counts[1]}"
        )

        self.logger.info("Dataset cleaned. Returning cleaned DataFrame.")
        return df

    def log_missing_values(self, df: pd.DataFrame) -> pd.Series:
        """
        Log missing values in a DataFrame.

        Process:
        1. Calculate the percentage of missing values for each column.
        2. Sort the columns by the percentage of missing values in
        descending order.
        3. Log the missing values.

        Parameters:
        ``df``: DataFrame to log missing values from

        Returns:
        ``missing_values``: Series containing the percentage of missing values
        for each column
        """

        if df.empty:
            logging.warning(
                "The DataFrame is empty. No missing values to log."
            )
            return

        self.logger.info("Logging missing values in the DataFrame")
        missing_values = df.isnull().sum() / len(df)

        self.logger.info(
            "Sorting missing values by percentage in descending order"
        )
        missing_values = missing_values[missing_values > 0]

        # Because this function is called in remove_missing_values, I won't
        # log the missing values here
        self.logger.info("Logging missing values")
        missing_values = missing_values.sort_values(ascending=False)

        return missing_values

    def remove_missing_values(self, df, threshold=0.3) -> pd.DataFrame:
        """
        Remove columns with missing values above a certain threshold.

        Process:

        1. Find missing values in the DataFrame.
        2. Remove columns with missing values above the threshold.
        3. Log the columns removed and return the updated DataFrame.

        Parameters:
        ``df``: DataFrame to remove missing values from
        ``threshold``: Threshold for missing values. Default is 0.3, because
        we're working with a large, financial dataset.

        Returns:
        ``df``: DataFrame with columns removed
        """

        self.logger.info("Finding missing values in the DataFrame")
        missing_values = self.log_missing_values(df)

        self.logger.info(
            "Removing columns with missing values above the threshold"
        )
        missing_values = missing_values[missing_values > threshold]
        df = df.drop(missing_values.index, axis=1)
        self.logger.info(
            "Columns removed. DataFrame updated. Returning updated DataFrame."
        )

        return df

    def determine_data_type(self, df: pd.DataFrame) -> tuple:
        """
        Determine whether a column is categorical or numerical.

        For a column to be cateogrical, it must either:
        1. Be of type object, or
        2. Have less than 20 unique values

        Process:
        1. Iterate through the columns in the DataFrame.
        2. Determine whether a column is categorical or numerical using the
        criteria above.
        3. Append the column to the appropriate list.

        Parameters:
        ``df``: DataFrame to determine data type for

        Returns:
        ``cat_cols``: List of categorical columns
        ``num_cols``: List of numerical columns
        """

        cat_cols = []
        num_cols = []

        self.logger.info(
            "Determining data type for each column in the DataFrame"
            )
        for col in df.columns:
            if df[col].dtype == "object" or df[col].nunique() < 20:
                cat_cols.append(col)
            else:
                num_cols.append(col)
        self.logger.info(
            f"There are {len(cat_cols)} categorical and {len(num_cols)} \
            numerical columns."
        )
        self.logger.info(
            "Data type determination complete. Returning categorical and \
            numerical columns."
        )

        return cat_cols, num_cols

    def balanced_sampling(self, df: pd.DataFrame):
        """
        Implement balanced sampling on the DataFrame.

        Process:
        1. Determine the target column.
        2. Determine the number of samples to take from each class.
        The number of samples taken will be the least frequent class.
        3. Sample the DataFrame and return the balanced DataFrame.

        Parameters:
        ``df``: DataFrame to sample

        Returns:
        ``df``: Balanced DataFrame
        """

        self.logger.info("Implementing balanced sampling on the DataFrame")
        target_col = self.target_col

        # The reason why I'm taking the least frequent class is because
        # I want to balance the classes - non-fraudulent transaction
        # over-representation is a problem in financial datasets.
        # This is my way of addressing it.
        self.logger.info(
            "Determining the number of samples to take from each class"
        )
        class_counts = df[target_col].value_counts()
        min_class_count = class_counts.min()
        sample_size = min_class_count

        self.logger.info("Sampling the DataFrame")
        sampled_df = (
            df.groupby(target_col)
            .apply(lambda x: x.sample(sample_size))
            .reset_index(drop=True)
        )

        self.logger.info(
            "Balanced sampling complete. Returning balanced DataFrame."
        )
        return sampled_df
