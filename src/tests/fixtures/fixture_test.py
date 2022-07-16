import pytest


@pytest.fixture
def preprocess () -> dict:
    """

    Returns
    -------
    dict
        _description_
    """

    customers = pd.read_csv("/workspaces/new_bpp/data/customer_data.csv")
    trans = pd.read_csv("/workspaces/new_bpp/data/transactions_data.csv")
    import datetime
    trans["Date"] = [
        datetime.datetime.strptime(date_, '%Y-%m-%d')
        for date_ in trans["Date"]
    ]
    complete_df = trans.merge(
        customers,
        on="Customer ID"
    ).drop("Loyalty Points", axis=1)

    vh = FeatureHandler(complete_df)
    vh.run_feat_buider()
    vh.categorical_to_numerical()

    mh = ModelHandler("/workspaces/new_bpp/config/config_trans.yaml")
    train_df, test_df, target, target_test = train_test_split(
        vh.df.loc[:, mh.features],
        vh.df["Incomplete Transaction"],
        test_size=0.20,
        random_state=77
    )

    return vh.df