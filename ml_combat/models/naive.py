
from .. import data


def fit(train):
    pass

def predict(test):
    res_dict = {}
    train = data.get_training_groupby_mean()
    train.head(1)
    for loc in train.index.get_level_values('location').unique():
        res_dict[loc] = train.loc[loc]['y', 'pv_measurement'].mean()

    test = data.get_testing()
    y_pred = test
    y_pred['y', 'NA', 'pv_measurement'] = 0

    for loc in train.index.get_level_values('location').unique():
        y_pred.loc[(loc, ), ('y', 'NA', 'pv_measurement')] = res_dict[loc]

    return y_pred[[['y', 'NA', 'pv_measurement']]]
    
def make_submittable(y_pred):
    y_pred = y_pred[[['y', 'NA', 'pv_measurement']]].reset_index()
    y_pred.columns = ['location', 'datetime', 'prediction']

    y_pred = y_pred[['prediction']].reset_index().rename(columns={'index': 'id'})
    return y_pred
