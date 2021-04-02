import pandas as pd
from typing import Dict


class CheckDataDtypes():
    def __init__(self, name: str=''):
        self.name_ = name
        self.data_dtypes = dict()

    def fit(self, data: pd.DataFrame, **kwargs):
        if not isinstance(data, pd.DataFrame):
            raise Exception('fit not a dataframe {}'.format(self.name_))
        self.data_dtypes = data.dtypes.astype(str).to_dict()
        return self
    
    def transform(self, data, **kwargs):
        if not isinstance(data, pd.DataFrame):
            raise Exception('transform not a dataframe {}'.format(self.name_))
        bad = []
        for k, v in self.data_dtypes.items():
            if v == data[k].dtype:
                continue
            try:
                data[k] = data[k].astype(v)
            except Exception as e:
                bad.append([k, data[k].dtype, v])
        if bad:
            raise Exception('bad data type {} {}'.format(self.name_, bad))
        return data


class CheckDataDictDtypes():
    def __init__(self):
        self.table_names_ = ()
        self.check_dtypes = dict()
        self.data_dtypes_ = dict()

    def fit(self, data_dict, **kwargs):
        self.table_names_ = tuple(data_dict.keys())
        self.check_dtypes = {name: CheckDataDtypes(name=name).fit(data) 
                                for name, data in data_dict.items()}
        self.data_dtypes_ = {name: self.check_dtypes[name].data_dtypes
                                for name in data_dict.keys()}
        return self

    def transform(self, data_dict, **kwargs):
        return  {name: self.check_dtypes[name].transform(data) 
                    for name, data in data_dict.items()}

if __name__ == "__main__":
    Data = {'Products': ['AAA','BBB','CCC','DDD','EEE'],
          'Prices': ['200','700','400','1200','900']}
    df = pd.DataFrame(Data)
    df2 = pd.DataFrame([[1, 2.3456, 'c', 'd', 78]], columns=list("ABCDE"))
    print (df)
    m = CheckDataDtypes('df').fit(df)
    print(m.data_dtypes)
    mm = CheckDataDictDtypes().fit({'1': df, '2': df2})
    print(mm.data_dtypes_)
    mm.transform({'1': df, '2': df2})
