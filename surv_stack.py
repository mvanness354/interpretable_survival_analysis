import numpy as np
import pandas as pd
from tqdm import tqdm

class SurvivalStacker:
    
    def __init__(self, discrete_time=True, sampling_ratio=1, keep_all_events=False):
        self.discrete_time = discrete_time
        self.sampling_ratio = sampling_ratio
        self.keep_all_events = keep_all_events
    
    def fit(self, times, censored):
        
        self.times = times
        self.censored = censored
        self.event_times = np.sort(times[~censored])
        return self
    
    
    def transform(self, X):
            
        data = None
        samples = []
        
        total_size = np.sum([
            np.sum(self.times >= event_time) - \
            int((1 - self.sampling_ratio) * np.sum(self.times > event_time)) \
            for event_time in self.event_times
        ])
        
        data = np.zeros((total_size, X.shape[1] + 2))
            
        
        count = 0
        i, j = 0, 0
        for event_time in tqdm(self.event_times):
            
            risk_set = self.times > event_time
            
            if self.keep_all_events:
                risk_set_keep = np.logical_and(
                    self.times > event_time,
                    ~self.censored
                )
                risk_set_sample = np.logical_and(
                    self.times > event_time,
                    self.censored
                )
            else:
                risk_set_keep = [False] * len(self.times)
                risk_set_sample = self.times > event_time
            
            if self.sampling_ratio < 1:
                risk_set_sub = np.random.choice(
                    risk_set_sample.nonzero()[0], 
                    size=int(len(risk_set.nonzero()[0]) * (1 - self.sampling_ratio)), 
                    replace=False
                )
                risk_set_sample[risk_set_sub] = False
                
            risk_set = np.logical_or.reduce((
                risk_set_keep,
                risk_set_sample,
                self.times == event_time,
            ))
            
            risk_set_samples = X[risk_set].values
            risk_set_samples = np.column_stack((
                risk_set_samples,
                np.repeat(event_time, risk_set_samples.shape[0]),
                np.logical_and(
                    self.times[risk_set] == event_time,
                    ~self.censored[risk_set]
                ).astype(int)
            ))
            
            n, p = risk_set_samples.shape
            data[i:(i+n), j:(j+p)] = risk_set_samples
            
            i += n

        df = pd.DataFrame(data, columns=list(X.columns) + ["time", "outcome"])
            
        
        if self.discrete_time:
            df["time"] = df["time"].astype('category')
            
        X = df.drop("outcome", axis=1)
        y = df["outcome"]
            
        return X, y
    
    def fit_transform(self, X, times, censored):
        self.fit(times, censored)
        return self.transform(X)
    
    def predict_survival_probs_(self, model, x, time, t_set):
        x_input = pd.DataFrame(
            np.tile(x.values.reshape(-1, 1), len(t_set)).T,
            columns=list(x.index)
        )
        x_input["time"] = t_set
        preds = model.predict_proba(x_input)[:, 1]
        return np.prod(1 - preds)
    
    def predict_survival_probs(self, model, X, time=5, pred_times=None, batch=False, batch_size=None):
    
        if pred_times is None:
            pred_times = self.times
            
        pred_times = pred_times[pred_times <= time]
    
        if not batch:
            tqdm.pandas()
            return X.progress_apply(
                lambda row: self.predict_survival_probs_(model, row, time, pred_times), 
                axis=1
            )

        else:
            t_set = self.event_times[self.event_times <= time]
            i = 0
            pred_surv = np.zeros(X.shape[0])
            for i in tqdm(range(0, X.shape[0], batch_size)):
                X_sub = X.iloc[i:(i+batch_size), :]
                X_stacked = X_sub.loc[np.repeat(X_sub.index, len(t_set))]
                X_stacked["time"] = np.tile(t_set, X_sub.shape[0])
                preds = model.predict_proba(X_stacked)[:, 1]
                pred_surv[i:(i+batch_size)] = (1 - preds).reshape(-1, len(t_set)).prod(axis=1)
                i += batch_size

            return np.array(preds_list)
        
    def predict_all_survival_probs(self, model, X, times=None):
        if times is None:
            times = self.times
        
        tqdm.pandas()
        return X.progress_apply(
            lambda row: self.predict_all_survival_probs_(model, row, times), 
            axis=1
        ).values
    
    def predict_all_survival_probs_(self, model, x, times):
        x_input = pd.DataFrame(
            np.tile(x.values.reshape(-1, 1), len(times)).T,
            columns=list(x.index)
        )
        x_input["time"] = times
        preds = model.predict_proba(x_input)[:, 1]
        return pd.Series(np.cumprod(1 - preds))
        
    
    def predict_cum_hazard_(self, model, x, time, t_set, monte_carlo):
        x_input = pd.DataFrame(
            np.tile(x.values.reshape(-1, 1), len(t_set)).T,
            columns=list(x.index)
        )
        x_input["time"] = t_set
        preds = model.predict_proba(x_input)[:, 1]
        
        if monte_carlo:
            return time * np.sum(preds) / len(t_set)
        else:
            return np.sum(preds)
    
    def predict_cum_hazard(
        self, model, X, time=5, 
        pred_times=None, batch=False, 
        batch_size=None, monte_carlo=True
    ):
        
        if pred_times is None:
            pred_times = self.times
            
        pred_times = pred_times[pred_times <= time]
        
        if not batch:
            tqdm.pandas()
            return X.progress_apply(
                lambda row: self.predict_cum_hazard_(model, row, time, pred_times, monte_carlo), 
                axis=1
            )
        
        else:
            t_set = self.event_times[self.event_times <= time]
            i = 0
            pred_cum_hazs = np.zeros(X.shape[0])
            for i in tqdm(range(0, X.shape[0], batch_size)):
                X_sub = X.iloc[i:(i+batch_size), :]
                X_stacked = X_sub.loc[np.repeat(X_sub.index, len(t_set))]
                X_stacked["time"] = np.tile(t_set, X_sub.shape[0])
                preds = model.predict_proba(X_stacked)[:, 1]
                pred_cum_hazs[i:(i+batch_size)] = preds.reshape(-1, len(t_set)).sum(axis=1)
                i += batch_size
                
            return np.array(preds_list)
        
    def predict_all_cum_hazard(self, model, X, times=None, monte_carlo=False):
        if times is None:
            times = self.times
        
        tqdm.pandas()
        return X.progress_apply(
            lambda row: self.predict_all_cum_hazard_(model, row, times, monte_carlo), 
            axis=1
        ).values
    
    def predict_all_cum_hazard_(self, model, x, times, monte_carlo=False):
        x_input = pd.DataFrame(
            np.tile(x.values.reshape(-1, 1), len(times)).T,
            columns=list(x.index)
        )
        x_input["time"] = times
        preds = model.predict_proba(x_input)[:, 1]
        
        if monte_carlo:
            return pd.Series(
                times * np.cumsum(preds) / np.arange(1, len(preds)+1)
            )
        else:
            return pd.Series(np.cumsum(preds))
        
            
        
            
        
        