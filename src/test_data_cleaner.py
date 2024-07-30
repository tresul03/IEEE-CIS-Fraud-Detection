import unittest
from data_cleaner import DataCleaner
import pandas as pd
import numpy as np

# ! This test is best run in parallel. Use pytest -n 5 to run in parallel


class TestDataCleaner(unittest.TestCase):
    def setUp(self):
        self.preprocessor = DataCleaner(target_col="target")
        self.preprocessor.logger.disabled = True

    def generate_synthetic_dataframe(
        self,
        n_samples: int = 1000,
        n_features: int = 4,
        nan_generation: bool = True
    ):
        """
        Generate a synthetic DataFrame for testing purposes.

        Process:
        1. Generate a DataFrame with normally distributed features.
        2. Introduce missing values in the DataFrame if nan_generation is True.

        Parameters:
        ``n_samples``: Number of samples in the DataFrame
        ``n_features``: Number of features in the DataFrame
        ``nan_generation``: Boolean to determine if missing values should be
        introduced in the DataFrame

        Returns:
        ``df``: Synthetic DataFrame
        ``column_nan_percentage_dict``: Dictionary containing the percentage of
        missing values for each column
        """

        np.random.seed(42)

        # Generate a DataFrame with the following features:
        # 1. Normally distributed numerical
        # 2. Categorical nominal and ordinal
        data = {
            f"feature_{i+1}": (
                np.random.normal(0, 1, n_samples)
                if i % 3 == 0  # Numerical
                else (
                    np.random.choice(["A", "B", "C"], n_samples)
                    if i % 3 == 1  # Categorical nominal
                    else np.random.randint(0, 15, n_samples)
                )
            )  # Categorical ordinal
            for i in range(n_features)
        }
        data["target"] = np.random.randint(0, 2, n_samples)
        df = pd.DataFrame(data)

        column_nan_percentages = {}

        if nan_generation:
            for column in df.columns[:-1]:  # Exclude the target column
                missing_percentage = np.random.uniform(0, 1.0)
                n_missing = int(n_samples * missing_percentage)

                column_nan_percentages[column] = missing_percentage

                missing_indices = np.random.choice(
                    df.index,
                    n_missing,
                    replace=False
                )

                df.loc[missing_indices, column] = np.nan

        # convert column_nan_percentage_dict to a Series
        column_nan_percentages = pd.Series(column_nan_percentages)

        return df, column_nan_percentages

    def test_remove_missing_values(self):
        """
        Test the remove_missing_values method in the Preprocessor class.

        Process:
        1. Generate a synthetic DataFrame with missing values.
        2. Remove missing values in the DataFrame using the method in question.
        3. Assert that the DataFrame has no missing values above 0.3.

        Returns:
        None
        """

        for _ in range(1000):
            n_samples = np.random.randint(100, 1000)
            n_features = np.random.randint(2, 20)

            with self.subTest(n_samples=n_samples, n_features=n_features):
                df = self.generate_synthetic_dataframe(
                    n_features=n_features,
                    n_samples=n_samples,
                    nan_generation=True
                )[0]
                df = self.preprocessor.remove_missing_values(
                    df,
                    threshold=0.3,
                )

                # Assert that the DataFrame has no missing values above 0.3
                missing_values = df.isnull().sum() / len(df)
                self.assertTrue(all(missing_values <= 0.3))

    def test_log_missing_values(self):
        """
        Test the log_missing_values method in the Preprocessor class.

        Process:
        1. Generate a synthetic DataFrame with missing values.
        2. Log the missing values in the DataFrame using the method in
        question.
        3. Assert that the missing values are equal to the percentage of
        missing values in the synthetic DataFrame.

        Returns:
        None
        """

        for _ in range(1000):
            n_samples = np.random.randint(100, 1000)
            n_features = np.random.randint(2, 20)

            with self.subTest(n_samples=n_samples, n_features=n_features):
                df, column_nan_dict = self.generate_synthetic_dataframe(
                    n_features=n_features,
                    n_samples=n_samples,
                    nan_generation=True
                )

                missing_values = self.preprocessor.log_missing_values(df)

                for column in missing_values.index:
                    self.assertAlmostEqual(
                        missing_values[column],
                        column_nan_dict[column],
                        places=1
                    )

    def test_determine_data_types(self):
        """
        Test the determine_data_type method in the Preprocessor class.

        Process:
        1. Generate a synthetic DataFrame with both numerical and categorical
        columns.
        2. Log the data types of the columns using the method in question.
        3. Assert that the columns are correctly classified as numerical or
        categorical.

        Returns:
        None
        """

        for _ in range(1000):
            n_samples = np.random.randint(100, 1000)
            n_features = np.random.randint(2, 20)

            with self.subTest(n_samples=n_samples, n_features=n_features):
                df, _ = self.generate_synthetic_dataframe(
                    n_features=n_features,
                    n_samples=n_samples,
                    nan_generation=False
                )

                df = self.preprocessor.remove_missing_values(df)

                X = df.drop(columns=["target"], axis=1)

                X_cat_cols, X_num_cols = \
                    self.preprocessor.determine_data_type(X)

                for column in X.columns:
                    if X[column].dtype == "object" or X[column].nunique() < 20:
                        self.assertIn(column, X_cat_cols)
                    else:
                        self.assertIn(column, X_num_cols)

    def test_balanced_sampling(self):
        """
        Test the balanced_sampling method in the Preprocessor class.

        Process:
        1. Generate a synthetic DataFrame with a target column.
        2. Implement balanced sampling on the DataFrame.
        3. Assert that the number of samples for each class is equal.

        Returns:
        None
        """

        for _ in range(1000):
            n_samples = np.random.randint(100, 1000)
            n_features = np.random.randint(2, 20)

            with self.subTest(n_samples=n_samples, n_features=n_features):
                df, _ = self.generate_synthetic_dataframe(
                    n_features=n_features,
                    n_samples=n_samples,
                    nan_generation=False
                )

                df = self.preprocessor.remove_missing_values(df)
                df = self.preprocessor.balanced_sampling(df)

                class_counts = df["target"].value_counts()

                self.assertEqual(class_counts[0], class_counts[1])

    def test_clean_dataset(self):
        """
        Test the clean_dataset method in the Preprocessor class.

        Process:
        1. Generate a synthetic DataFrame with missing values.
        2. Clean the DataFrame using the method in question.
        3. Assert that the DataFrame has no missing values above 0.3.
        4. Assert that the number of samples for each class is equal.

        Returns:
        None
        """

        for _ in range(1000):
            n_samples = np.random.randint(100, 1000)
            n_features = np.random.randint(2, 20)

            with self.subTest(n_samples=n_samples, n_features=n_features):
                df, _ = self.generate_synthetic_dataframe(
                    n_features=n_features,
                    n_samples=n_samples,
                    nan_generation=True
                )

                # Assert the following:
                # 1. No columns have nan percentages above 0.3
                # 2. The number of samples for each class is equal
                df = self.preprocessor.clean_dataset(df)
                missing_values = df.isnull().sum() / len(df)
                class_counts = df["target"].value_counts()

                self.assertTrue(all(missing_values <= 0.3))
                self.assertEqual(class_counts[0], class_counts[1])


if __name__ == "__main__":
    unittest.main()
