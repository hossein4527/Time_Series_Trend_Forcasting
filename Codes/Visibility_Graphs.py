import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from networkx.linalg.graphmatrix import adjacency_matrix
import math
from visibility_graph import visibility_graph

class VG_Modules_Model1(object):
    def __init__(self):
        self.bw = 1
        
    #Usual Model One Prediction functions
    def most_similar_node(self,time_series, T=20, tp=30):
        G = visibility_graph(time_series)
        adj_matrix = np.array(adjacency_matrix(G).todense())
        E = G.number_of_edges()
        k = adj_matrix.sum(1)
        P = adj_matrix/k
        n = k.shape[0]
        s_srw = np.zeros((n,n))
        pi = np.diag(np.ones(n))
        for t in range(T):
            pi = P@pi
            s_srw += (k*pi+k.T*pi.T)/(2*E)
        if tp != 0:
            triup = np.triu(s_srw,k=-tp)
            triup[triup != 0] = 1
            s_srw = np.tril(s_srw,k=-1) * triup
        return np.argmax(s_srw,axis=1)

        
    def ts_forcast_model1(self,time_serie):
        start_point = 31
        forcast_ts = np.zeros_like(time_serie)
        forcast_ts[:start_point] = time_serie[:start_point]
        msn = self.most_similar_node(time_serie,T=20,tp=50)
        for i in range(start_point-1,time_serie.shape[0]-1):
            dy = time_serie[i]-time_serie[msn[i]]
            y = time_serie[i]+dy/(i-msn[i])
            yhat = (i-msn[i])/(i-msn[i]+1)*y+(1/(i-msn[i]+1))*time_serie[i]
            forcast_ts[i+1]=yhat
        return forcast_ts
    
    #Prediction Model One Based on 2 Regimed Data
    def most_similar_matrix(self, time_series, T=20, tp=30):
        G = visibility_graph(time_series)
        adj_matrix = np.array(adjacency_matrix(G).todense())
        E = G.number_of_edges()
        k = adj_matrix.sum(1)
        P = adj_matrix/k
        n = k.shape[0]
        s_srw = np.zeros((n,n))
        pi = np.diag(np.ones(n))
        for t in range(T):
            pi = P@pi
            s_srw += (k*pi+k.T*pi.T)/(2*E)
        if tp != 0:
            triup = np.triu(s_srw,k=-tp)
            triup[triup != 0] = 1
            s_srw = np.tril(s_srw,k=-1) * triup
        return s_srw

    def selective_msn_2regimed_Ox(self, srw, df, tp=20):
        indx = []
        rg_3 = []
        ms_reg = []
        for k in range(len(srw)):
            indx.append([])
            rg_3.append([])
            rg_3[-1].append(df.iloc[k]['Ox_2'])
            for i in range(1,tp):
                indx[-1].append(np.where(srw[k] == np.sort(srw[k])[-i])[0][0])
                rg_3[-1].append(df.iloc[indx[-1][-1]]['Ox_2'])
            try:
                ms_reg.append(indx[-1][np.where(rg_3[-1]==rg_3[-1][0])[0][1]])
            except IndexError:
                ms_reg.append(indx[-1][1])
        return ms_reg

    
    def ts_forcast_2Regimed_model1(self, time_serie, df):
        start_point = 31
        forcast_ts = np.zeros_like(time_serie)
        forcast_ts[:start_point] = time_serie[:start_point]
        msns = self.most_similar_matrix(time_serie,T=20,tp=50)
        msn = self.selective_msn_2regimed_Ox(msns, df)
        for i in range(start_point-1,time_serie.shape[0]-1):
            dy = time_serie[i]-time_serie[msn[i]]
            y = time_serie[i]+dy/(i-msn[i])
            yhat = (i-msn[i])/(i-msn[i]+1)*y+(1/(i-msn[i]+1))*time_serie[i]
            forcast_ts[i+1]=yhat
        return forcast_ts
    
    #Prediction Model One Based on 12 Regimed Data
    def selective_msn_12regmied(self, srw, df, tp=20):
        indx = []
        rg_3 = []
        ms_reg = []
        for k in range(len(srw)):
            indx.append([])
            rg_3.append([])
            rg_3[-1].append(df.iloc[k]['12 today/tommorrow'])
            for i in range(1,tp):
                indx[-1].append(np.where(srw[k] == np.sort(srw[k])[-i])[0][0])
                rg_3[-1].append(df.iloc[indx[-1][-1]]['12 today/tommorrow'])
            try:
                ms_reg.append(indx[-1][np.where(rg_3[-1]==rg_3[-1][0])[0][1]])
            except IndexError:
                ms_reg.append(indx[-1][1])
        return ms_reg
    
    def ts_forcast_12Regimed_model1(self, time_serie, df):
        start_point = 31
        forcast_ts = np.zeros_like(time_serie)
        forcast_ts[:start_point] = time_serie[:start_point]
        msns = self.most_similar_matrix(time_serie,T=20,tp=50)
        msn = self.selective_msn_12regmied(msns, df)
        for i in range(start_point-1,time_serie.shape[0]-1):
            dy = time_serie[i]-time_serie[msn[i]]
            y = time_serie[i]+dy/(i-msn[i])
            yhat = (i-msn[i])/(i-msn[i]+1)*y+(1/(i-msn[i]+1))*time_serie[i]
            forcast_ts[i+1]=yhat
        return forcast_ts
    
    def error_bars(self, selected_data, forcast_ts):
        RMSE = math.sqrt(np.square(np.subtract(selected_data,forcast_ts)).mean())
        NRMSE = RMSE/(max(selected_data)-min(selected_data))
        MAPE=np.mean(np.abs((selected_data - forcast_ts)/selected_data))*100
        MAP=np.mean(np.abs((selected_data - forcast_ts)))
        return RMSE, NRMSE, MAPE, MAP

    #Prediction Model One Based on 3 Regimed Data
    def selective_msn_3regmied(self, srw, df, tp=20):
        indx = []
        rg_3 = []
        ms_reg = []
        for k in range(len(srw)):
            indx.append([])
            rg_3.append([])
            rg_3[-1].append(df.iloc[k]['Reg_3'])
            for i in range(1,tp):
                indx[-1].append(np.where(srw[k] == np.sort(srw[k])[-i])[0][0])
                rg_3[-1].append(df.iloc[indx[-1][-1]]['Reg_3'])
            try:
                ms_reg.append(indx[-1][np.where(rg_3[-1]==rg_3[-1][0])[0][1]])
            except IndexError:
                ms_reg.append(indx[-1][1])
        return ms_reg
    
    def ts_forcast_3Regimed_model1(self, time_serie, df):
        start_point = 31
        forcast_ts = np.zeros_like(time_serie)
        forcast_ts[:start_point] = time_serie[:start_point]
        msns = self.most_similar_matrix(time_serie,T=20,tp=50)
        msn = self.selective_msn_3regmied(msns, df)
        for i in range(start_point-1,time_serie.shape[0]-1):
            dy = time_serie[i]-time_serie[msn[i]]
            y = time_serie[i]+dy/(i-msn[i])
            yhat = (i-msn[i])/(i-msn[i]+1)*y+(1/(i-msn[i]+1))*time_serie[i]
            forcast_ts[i+1]=yhat
        return forcast_ts
    
    def error_bars(self, selected_data, forcast_ts):
        RMSE = math.sqrt(np.square(np.subtract(selected_data,forcast_ts)).mean())
        NRMSE = RMSE/(max(selected_data)-min(selected_data))
        MAPE=np.mean(np.abs((selected_data - forcast_ts)/selected_data))*100
        MAP=np.mean(np.abs((selected_data - forcast_ts)))
        return RMSE, NRMSE, MAPE, MAP
    

class VG_Modules_Model2(object):
    def __init__(self):
        self.bw = 1
    
    #Usual Model Two Prediction functions
    def node_similarity(self, time_series, T=20, tp=50):
        G = visibility_graph(time_series)
        adj_matrix = np.array(adjacency_matrix(G).todense())
        E = G.number_of_edges()
        k = adj_matrix.sum(1)
        P = adj_matrix/k
        n = k.shape[0]
        s_srw = np.zeros((n,n))
        pi = np.diag(np.ones(n))
        for t in range(T):
            pi = P@pi
            s_srw += (k*pi+k.T*pi.T)/(2*E)
        if tp != 0:
            triup = np.triu(s_srw,k=-tp)
            triup[triup != 0] = 1
            s_srw = np.tril(s_srw,k=-1) * triup
        return np.argmax(s_srw,axis=1) , s_srw
    
    def ts_forcast_model2_weighted(self,time_serie, tp=30):
        arg , s_srw = self.node_similarity(time_serie,T=20,tp=30)
        sum_s = s_srw.sum(axis=1)
        sum_s[sum_s == 0] = np.inf
        ws_m = (s_srw.T/sum_s).T
        ws = np.zeros((ws_m.shape[0], tp))
        for i in range(ws_m.shape[0]):
            ws[i] = np.roll(ws_m[i], tp-i)[:tp]
        start_point = 31
        forcast_ts = np.zeros_like(time_serie)
        forcast_ts[:start_point] = time_serie[:start_point]
        for i in range(start_point-1,time_serie.shape[0]-1):
            y = time_serie[i]+(time_serie[i]-time_serie[i-tp:i])/(i-np.arange(i-tp,i,1))
            forcast_ts[i+1]=(ws[i]*y).sum()
        
        return forcast_ts
    
    #Prediction Model Two Based on 2 Regimed Data
    def most_similarity_matrix(self,time_series, T=20, tp=50):
        G = visibility_graph(time_series)
        adj_matrix = np.array(adjacency_matrix(G).todense())
        E = G.number_of_edges()
        k = adj_matrix.sum(1)
        P = adj_matrix/k
        n = k.shape[0]
        s_srw = np.zeros((n,n))
        pi = np.diag(np.ones(n))
        for t in range(T):
            pi = P@pi
            s_srw += (k*pi+k.T*pi.T)/(2*E)
        if tp != 0:
            triup = np.triu(s_srw,k=-tp)
            triup[triup != 0] = 1
            s_srw = np.tril(s_srw,k=-1) * triup
        return s_srw
    
    def selective_msn_2Regimed_Ox_model2_weighted(self,srw,df,  tp=20):
        indx = []
        rg_3 = []
        ms_reg = []
        for k in range(len(srw)):
            indx.append([])
            rg_3.append([])
            rg_3[-1].append(df.iloc[k]['Ox_2'])
            for i in range(1,tp):
                indx[-1].append(np.where(srw[k] == np.sort(srw[k])[-i])[0][0])
                rg_3[-1].append(df.iloc[indx[-1][-1]]['Ox_2'])
            try:
                ms_reg.append(indx[-1][np.where(rg_3[-1]==rg_3[-1][0])[0][1]])
            except IndexError:
                ms_reg.append(indx[-1][1])
        return ms_reg
    
    def ts_forcast_2Regimed_model2_weighted(self, time_serie, df):
        start_point = 31
        forcast_ts = np.zeros_like(time_serie)
        forcast_ts[:start_point] = time_serie[:start_point]
        msns = self.most_similarity_matrix(time_serie,T=20,tp=50)
        msn = self.selective_msn_2Regimed_Ox_model2_weighted(msns, df)
        for i in range(start_point-1,time_serie.shape[0]-1):
            dy = time_serie[i]-time_serie[msn[i]]
            y = time_serie[i]+dy/(i-msn[i])
            yhat = (i-msn[i])/(i-msn[i]+1)*y+(1/(i-msn[i]+1))*time_serie[i]
            forcast_ts[i+1]=yhat
        return forcast_ts
    
    #Prediction Model Two Based on 3 Regimed Data
    def most_similarity_matrix(self,time_series, T=20, tp=50):
        G = visibility_graph(time_series)
        adj_matrix = np.array(adjacency_matrix(G).todense())
        E = G.number_of_edges()
        k = adj_matrix.sum(1)
        P = adj_matrix/k
        n = k.shape[0]
        s_srw = np.zeros((n,n))
        pi = np.diag(np.ones(n))
        for t in range(T):
            pi = P@pi
            s_srw += (k*pi+k.T*pi.T)/(2*E)
        if tp != 0:
            triup = np.triu(s_srw,k=-tp)
            triup[triup != 0] = 1
            s_srw = np.tril(s_srw,k=-1) * triup
        return s_srw
    
    def selective_msn_3Regimed_Ox_model2_weighted(self,srw,df,  tp=20):
        indx = []
        rg_3 = []
        ms_reg = []
        for k in range(len(srw)):
            indx.append([])
            rg_3.append([])
            rg_3[-1].append(df.iloc[k]['Reg_3'])
            for i in range(1,tp):
                indx[-1].append(np.where(srw[k] == np.sort(srw[k])[-i])[0][0])
                rg_3[-1].append(df.iloc[indx[-1][-1]]['Reg_3'])
            try:
                ms_reg.append(indx[-1][np.where(rg_3[-1]==rg_3[-1][0])[0][1]])
            except IndexError:
                ms_reg.append(indx[-1][1])
        return ms_reg
    
    def ts_forcast_3Regimed_model2_weighted(self, time_serie, df):
        start_point = 31
        forcast_ts = np.zeros_like(time_serie)
        forcast_ts[:start_point] = time_serie[:start_point]
        msns = self.most_similarity_matrix(time_serie,T=20,tp=50)
        msn = self.selective_msn_3Regimed_Ox_model2_weighted(msns, df)
        for i in range(start_point-1,time_serie.shape[0]-1):
            dy = time_serie[i]-time_serie[msn[i]]
            y = time_serie[i]+dy/(i-msn[i])
            yhat = (i-msn[i])/(i-msn[i]+1)*y+(1/(i-msn[i]+1))*time_serie[i]
            forcast_ts[i+1]=yhat
        return forcast_ts
    
    def ts_forcast_3Regimed_weighted_compact(self, df , time_serie):
        s_srw = self.most_similarity_matrix(time_serie, T=20,tp=50)
        sum_s = s_srw.sum(axis=1)
        sum_s[sum_s == 0] = np.inf
        ws_m = (s_srw.T/sum_s).T
        ws = np.zeros((ws_m.shape[0], tp))
        
        srw = s_srw
        indx = []
        rg_3 = []
        ms_reg = []
        total_y_indexs = []
        total_y_srws = []
        tp=50
        for k in range(len(srw)):
            total_y_indexs.append([])
            total_y_srws.append([])
            indx.append([])
            # indx[-1].append(0)
            rg_3.append([])
            rg_3[-1].append(df.iloc[k]['3_regime_index'])
            for i in range(tp):
                indx[-1].append(np.where(srw[k] == np.sort(srw[k])[::-1][:tp][i])[0][0])
                rg_3[-1].append(df.iloc[indx[-1][-1]]['3_regime_index'])
                
            if rg_3[k][0] != 2:
                same_reg = np.where(rg_3[k]==rg_3[k][0])[0]
                turn_points = np.where(np.array(rg_3[k])==2)[0]
                num_same_regime = len(same_reg)
                num_turn_ponits = len(turn_points)
                indx_same = np.array(indx[k])[list(same_reg[1:]-1)]
                indx_2 = np.array(indx[k])[list(turn_points-1)]
                total_y_indexs[-1].append(np.concatenate((indx_same,indx_2), axis=0))
                srw[k][indx_2] *= num_turn_ponits
                srw[k][indx_same] *= num_same_regime
                ssr = srw[k][total_y_indexs[-1]]
                
            
                total_y_srws[-1].append(ssr[0]/sum(ssr[0]))
                
            if rg_3[k][0] == 2:
                same_reg = np.where(rg_3[k]==rg_3[k][0])[0]
                other_points = np.where(np.array(rg_3[k])!=2)[0]
                num_same_regime = len(same_reg)
                num_turn_ponits = len(other_points)
                indx_same = np.array(indx[k])[list(same_reg[1:]-1)]
                indx_2 = np.array(indx[k])[list(other_points-1)]
                total_y_indexs[-1].append(np.concatenate((indx_same,indx_2), axis=0))
                
                srw[k][indx_2] *= num_turn_ponits
                srw[k][indx_same] *= num_same_regime 
                
                ssr = srw[k][total_y_indexs[-1]]
                # ssr_0 , ssr_1 = np.unique(ssr[0], return_index=True)
                # ssr = ssr_0[np.argsort(ssr_1)]
                # ssr = ssr[ssr!=0]
                # total_y_indexs = total_y_indexs[np.argsort(ssr_1)]
                
                total_y_srws[-1].append(ssr[0]/sum(ssr[0]))
                
                
        start_point = 31
        forcast_ts = np.zeros_like(time_serie)
        forcast_ts[:start_point] = time_serie[:start_point]
        for i in range(start_point-1,time_serie.shape[0]-1):
            y = time_serie[i]+(time_serie[i]-time_serie[total_y_indexs[i][0]])/(i-total_y_indexs[i][0])
            forcast_ts[i+1]=(total_y_srws[i]*y).sum()
            
        return forcast_ts
    
    def error_bars(self, selected_data, forcast_ts):
        RMSE = math.sqrt(np.square(np.subtract(selected_data,forcast_ts)).mean())
        NRMSE = RMSE/(max(selected_data)-min(selected_data))
        MAPE=np.mean(np.abs((selected_data - forcast_ts)/selected_data))*100
        MAP=np.mean(np.abs((selected_data - forcast_ts)))
        return RMSE, NRMSE, MAPE, MAP

                    
