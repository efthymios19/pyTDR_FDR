import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np
from numpy import isnan
import math 
from datetime import datetime, timedelta

class Todoroff:
    def __init__(self,file):
        self.file=file
    def import_csv(self):
        if (not self.file.endswith('.csv')):
            file = self.file + ".csv"
        else:
            file = self.file
        df = pd.read_csv(file,skiprows=0)
        #df=df.drop([0,1])
        unique_timestamps=df["TIMESTAMP"].nunique()
        if unique_timestamps ==1:
            timestamp = df.loc[2]["TIMESTAMP"]
            date, time = timestamp.split(" ")
            Vp=df.loc[2]["Vp"]
            WaveAvg=df.loc[2]["Averages"]
            points=df.loc[2]["Points"]
            cablelen=df.loc[2]["CableLen"]
            windowlen=df.loc[2]["WindowLen"]
            probelen=df.loc[2]["ProbeLen"]
            Vp=float(Vp)
            points=int(points)
            windowlen=int(windowlen)
            probelen=float(probelen)
            measurements = df.loc[2]["Waveform(1)":] # extract measurements from Waveform 1 row
            measurements = measurements.astype(float)  # convert to float type
            measurements = measurements[~isnan(measurements)]  # filter out NaN values
            measurements.iloc[0]=0 #setting the beginning of the trace to zero
        else:
            timestamp = df["TIMESTAMP"]
            Vp=df["Vp"]
            WaveAvg=df["AverageCnt"]
            points=df["PointsCnt"]
            cablelen=df["CableLen"]
            windowlen=df["WindowLen"]
            probelen=df["ProbeLen"]
            Vp=Vp.astype(float)
            points=points.astype(int)
            windowlen=windowlen.astype(int)
            probelen=probelen.astype(float)
            selected_columns=[]
            for i in range(1,points.loc[2]+1):
                selected_columns.append(f"Waveform({i})")
            measurements=df[selected_columns].copy()
            measurements = measurements.astype(float)  # convert to float type
            measurements = measurements[~isnan(measurements)]  # filter out NaN values
            measurements["Waveform(1)"]=0
        return timestamp, Vp, WaveAvg, points, cablelen, windowlen, probelen, measurements
    
    def impulse_response(self):
        timestamp, Vp, WaveAvg, points, cablelen, windowlen, probelen, measurements=self.import_csv()
        self.LightSpeed=3*(10**8)
        #Vp=Vp.astype(float)
        #points=points.astype(int)
        #windowlen=windowlen.astype(float)
        num_rows = points.shape[0]
        for idx in points.index:
            num_points=points.loc[idx]
            WaveformApparentDistance=np.zeros((num_rows,num_points))
            WaveformResponseTime=np.zeros((num_rows,num_points))
            ImpulseResponse=np.zeros((num_rows,num_points))
            Reflectedsignal=measurements.values
            VDown = [np.zeros((num_points, num_points)) for _ in range(num_rows)]
            for j in range(num_rows):
                WaveformApparentDistance[j,0]=0
                ImpulseResponse[j,0]=0
                for i in range(1, num_points):
                    WaveformApparentDistance[j,i] = WaveformApparentDistance[j,i - 1] + (windowlen.loc[idx] / (points.loc[idx] - 1))
                    WaveformResponseTime[j,i] = (2 * WaveformApparentDistance[j,i] / (self.LightSpeed * Vp.loc[idx]))*1000000000
                for i in range(0, num_points):
                    Reflectedsignal[j,i]=Reflectedsignal[j,i]-1
                for i in range(1, num_points-1):
                    ImpulseResponse[j,i] = Reflectedsignal[j,i+1]-Reflectedsignal[j,i]
                for i in range(0, num_points):
                    VDown[j][i, 0] = ImpulseResponse[j,i]
                    
        return WaveformApparentDistance, WaveformResponseTime, ImpulseResponse, VDown

    def Selectstartendprobe(self):
        timestamp, Vp, WaveAvg, points, cablelen, windowlen, probelen, measurements=self.import_csv()
        WaveformApparentDistance, WaveformResponseTime, ImpulseResponse, VDown=self.impulse_response()
        num_rows = points.shape[0]
        for idx in points.index:
            num_points=points.loc[idx]
            Segments=np.zeros((num_rows, num_points))
            dy_dt=np.zeros((num_rows,num_points))
            dy_dt2=np.zeros((num_rows,num_points))
            dy_dt_alt=np.zeros((num_rows,num_points))
            measur=measurements.values
            #points=points.astype(int)
            for j in range(num_rows):
                Segments[j]=np.arange(1,num_points+1)
                for i in range(0, len(WaveformResponseTime[j])-1):
                    dy_dt[j,i]=((measur[j,i+1]-measur[j,i])/(WaveformResponseTime[j,i+1]-WaveformResponseTime[j,i]))
                dy_dt[j,num_points-1]=-measur[j,num_points-1]/-WaveformResponseTime[j,num_points-1]
                for i in range(0, len(WaveformResponseTime[j])-1):
                    dy_dt2=(dy_dt[j,i+1]-dy_dt[j,i])/(WaveformResponseTime[j,i+1]-WaveformResponseTime[j,i])
                #dy_dt2[j,num_points-1]=-dy_dt[j,num_points-1]/-WaveformResponseTime[j,num_points-1]
                for i in range(0, len(WaveformResponseTime[j])-2):
                    dy_dt_alt[j,i]=(measur[j,i+2]-measur[j,i])/(WaveformResponseTime[j,i+2]-WaveformResponseTime[j,i])
                #dy_dt_alt[j,num_points-2]=(-measurements[j,points-1]/-WaveformResponseTime[j,points-1])
        peak_indices = []
        # Iterate over each array dy_dt[j]
        for j in range(len(dy_dt)):
            # Get the indices of the 20 highest peaks in dy_dt[j]
            indices = sorted(range(len(dy_dt[j])), key=dy_dt[j].__getitem__, reverse=True)[:20]
            peak_indices.append(indices)
      
        return Segments,dy_dt, dy_dt2, dy_dt_alt, peak_indices
            
    def forwardmodel(self):
        timestamp, Vp, WaveAvg, points, cablelen, windowlen, probelen, measurements=self.import_csv()
        WaveformApparentDistance, WaveformResponseTime, ImpulseResponse, VDown=self.impulse_response()
        #points=points.astype(int)
        num_rows = points.shape[0]
        for idx in points.index:
            num_points=points.loc[idx]
            Vup = [np.zeros((num_points, num_points)) for _ in range(num_rows)]
            Segmentreflcoef=[np.zeros((1,num_points)) for _ in range(num_rows)]
        for x in range(num_rows):
            Vup[x][0,0]=1
            Segmentreflcoef[x][0,0]=VDown[x][1,0]/Vup[x][0,0]
            Vup[x][0,1]=Vup[x][0,0]*(1+Segmentreflcoef[x][0,0])
            for j in range(1, num_points-1):
                VDown[x][j,1]=VDown[x][j+1,0]/(1-Segmentreflcoef[x][0,0])
                Vup[x][j,1]=-VDown[x][j,1]*Segmentreflcoef[x][0,0]
            Segmentreflcoef[x][0,1]=VDown[x][1,1]/Vup[x][0,1]
            Vup[x][0,2]= Vup[x][0,1]*(1+Segmentreflcoef[x][0,1])
            
            for i in range(2, num_points-1):
                for j in range(1, num_points-i):
                    VDown[x][j,i]= (VDown[x][j+1, i-1] - Segmentreflcoef[x][0, i-1]*Vup[x][j,i-1])/ (1-Segmentreflcoef[x][0,i-1])
                    Vup[x][i,j]= (1+Segmentreflcoef[x][0,i-1])*Vup[x][j,i-1] - Segmentreflcoef[x][0,i-1]*VDown[x][j,i]
                Segmentreflcoef[x][0,i]=VDown[x][1,i]/Vup[x][0,i]
                Vup[x][0,i+1]=Vup[x][0,i]*(1+Segmentreflcoef[x][0,i])

            original_values = np.cumsum(Segmentreflcoef[x])

        return Segmentreflcoef, original_values, VDown, Vup
    
    def inversemodel(self, idxfirst, idxsec, ProbeairImp=140):
        """
        idxfirst: The index of the first peak (start of the probe)
        idsxsec: The index of the second peak (end of the probe)
        ProbeairImp: Impendance of the probe in the air

        """
        timestamp, Vp, WaveAvg, points, cablelen, windowlen, probelen, measurements=self.import_csv()
        WaveformApparentDistance, WaveformResponseTime, ImpulseResponse, VDown=self.impulse_response()
        Segmentreflcoef, original_values, VDown, Vup=self.forwardmodel()
        ProbeairImp=140
        num_rows = points.shape[0]
        self.Ka_all=[]
        self.lenact_all=[]
        for idx in points.index:
            num_points=points.loc[idx]
            Impendance=[np.zeros((1,num_points)) for _ in range(num_rows)]
            DielectricConstant=np.zeros((num_rows,num_points))
            Actuallength=np.zeros((num_rows,num_points))
            length=np.zeros((num_rows,num_points))
            for j in range(num_rows):
                Traveltime=(WaveformResponseTime[j,2]-WaveformResponseTime[j,1])/1000000000
                Impendance[j][0,0]=50
                for i in range(1, num_points):
                    Impendance[j][0,i]=Impendance[j][0,i-1]*((1+Segmentreflcoef[j][0,i-1])/(1-Segmentreflcoef[j][0,i-1]))
                for i in range(idxfirst,idxsec+1):
                    DielectricConstant[j,i]=(ProbeairImp/Impendance[j][0,i])**2
                    Actuallength[j,i]=(Traveltime*self.LightSpeed)/(2*(DielectricConstant[j,i]**0.5))
                length[j,0]=Actuallength[j,0]
                for i in range(1,len(Actuallength[j])-1):
                    length[j,i]=Actuallength[j,i]+length[j,i-1]

        for j in range(num_rows):
            lenact=length[j][idxfirst-1:idxsec+1]
            Ka=DielectricConstant[j][idxfirst-1:idxsec+1]
            self.lenact_all.append(lenact)
            self.Ka_all.append(Ka)

        self.lenact_all = np.array(self.lenact_all)
        self.Ka_all = np.array(self.Ka_all)
        return Impendance, self.Ka_all, self.lenact_all
    
    def calibratewatercontent(self, idxfirst, idxsec, node_depths=None, timeframe='3H'):
        """
        idxfirst: The index of the first peak (start of the probe)
        idsxsec: The index of the second peak (end of the probe)
        node_depths: A list with the depth of installed point soil moisture sensors (m.)
        timeframe: The timestep of the measuremnts with spatial TDR
        """
        timestamp, Vp, WaveAvg, points, cablelen, windowlen, probelen, measurements = self.import_csv()
        Impendance, Ka, lenact=self.inversemodel(idxfirst,idxsec)
        node_dataframes={} #dictionary to store dataframes
        for depth in node_depths:
            file_path=input(f"Please enter the .csv file path for the node at depth {depth} m.")
            # Read the CSV file into a DataFrame and store it in the dictionary
            try:
                df = pd.read_csv(file_path)  # Read the CSV file into a DataFrame
                node_dataframes[f"soilmoist_{depth}"] = df  # Store the DataFrame with a key indicating its depth
                print(f"Successfully imported data for depth {depth} m.")
            except Exception as e:
                print(f"Error importing data from {file_path}: {e}")
            
        start_date=timestamp.iloc[0]
        timestamp=pd.to_datetime(timestamp)
        end_date=timestamp.iloc[-1] + timedelta(hours=1)
        end_date=end_date.strftime('%Y-%m-%d %H:%M:%S')
        soilmoistarrays=[]
        for depth, df in node_dataframes.items():
            #Filter the Datataframe within the start and the end data
            filtered_df=df[(df['Time']>=start_date) & (df['Time']<=end_date)]
            node_dataframes[depth]=filtered_df
            globals()[f"soilmoist_{depth}m"] = filtered_df 
            #Convert 'Time' column to datetime format and set as index
            df['Time']=pd.to_datetime(df['Time'], format='mixed')
            df=df.set_index('Time')
            # Resample the data to 3 - hour interval
            df_3hr=df.resample(timeframe).mean()
            # Convert timestamp to a set of datetimes for quick filtering
            df_3hr.index=df_3hr.index.tz_localize(None).astype('datetime64[ns]')
            # Filter dataframes to only keep rows where the index is in the timestamp set
            df_3hr = df_3hr[df_3hr.index.isin(timestamp)]
            # Extract 'water_SOIL' values as an array for further processing
            water_soil_array = df_3hr['water_SOIL'].values
            soilmoistarrays.append(water_soil_array)
            # Optionally store the resampled and filtered DataFrame and array back in node_dataframes
            node_dataframes[depth] = df_3hr  # Update the original dictionary with filtered/resampled data
            globals()[f"{depth}_3hr_array"] = water_soil_array
            
        mean_Ka_dict={depth: [] for depth in node_depths}
        for timestep in range(len(timestamp)):
            current_time=timestamp[timestep]

            for depth in node_depths:
                #Find indices within a small range around the target depth
                indices=np.where((lenact[timestep] >= depth - 0.05) & (lenact[timestep] < depth + 0.05))[0]
                # Calculate the mean Ka for the current depth if indices are found
                mean_Ka = np.mean(Ka[timestep][indices]) if indices.size > 0 else np.nan
                mean_Ka_dict[depth].append(mean_Ka)  # Append mean Ka value to the list for this depth

        # Convert lists in mean_Ka_dict to NumPy arrays after loop completes
        mean_Ka_dict = {depth: np.array(mean_values) for depth, mean_values in mean_Ka_dict.items()}
        X=tuple(mean_Ka_dict.values())
        y=np.concatenate(soilmoistarrays)
        #Modified Topp model:
        def ModTopp(X,d,c,b,a):
            x1, x2 = X
            y1=d+ c*x1 + b*(x1**2)+a*(x1**3)
            y2=d+ c*x2 + b*(x2**2)+a*(x2**3)
            return np.concatenate([y1, y2])
        initial_guesses=[-5.3e-02, 2.92e-02, -5.5e-04, 4.3e-06]
        popt_Topp,pcov_Topp=curve_fit(ModTopp, X,y,p0=initial_guesses,maxfev=100000000)
        
        return node_dataframes, popt_Topp, mean_Ka_dict, soilmoistarrays
    
    def watercontentcomp(self, model='Topp', Topp_params=None):
        """
        model: string, 
        1.Topp
        2. Modified Topp (ModTopp)
        Topp_params: List of Topp params produced by calibratewatercontent
        
        """
        num_rows = self.Ka_all.shape[0]
        for i, row in enumerate(self.Ka_all):
              a=len(row)
        self.thetatop=np.zeros((num_rows , a))
        if model=="Topp":
            for i in range(0, len(self.Ka_all)):
                 self.thetatop[i] = -5.3e-02 + 2.92e-02 * self.Ka_all[i] + (-5.5e-04) * (self.Ka_all[i] ** 2) + 4.3e-06 * (self.Ka_all[i] ** 3)
        if model=="ModTopp":
            for i in range(0, len(self.Ka_all)):
                self.thetatop[i]=Topp_params[0] + Topp_params[1] * self.Ka_all[i] + Topp_params[2] * (self.Ka_all[i] ** 2) + Topp_params[3] * (self.Ka_all[i] **3)
        return self.thetatop

    def calculate_mean_wc_at_depths(self, target_depths):
        """
        Calculate the mean water content (WC) at the specified target depths for each timestep.
        Parameters:
        - target_depths: List of depths (in meters) at which to calculate the mean WC.
        Returns:
        - mean_wc_dict: A dictionary where keys are depth names (e.g., '0.1m', '0.5m') and values are lists of mean WC values at those depths.
        """
        timestamp, Vp, WaveAvg, points, cablelen, windowlen, probelen, measurements = self.import_csv()
        # Initialize a dictionary to store mean WC values for each depth
        self.mean_wc_dict = {f"{depth}m": [] for depth in target_depths}
        
        # Loop through each timestep
        for timestep in range(len(timestamp)):
            
            # Get the actual timestamp for the current timestep
            current_time = timestamp[timestep]
            # Loop through each target depth
            for depth in target_depths:
                # Find indices where lenact values are within a small range around the target depth
                indices = np.where((self.lenact_all[timestep] >= depth - 0.05) & (self.lenact_all[timestep] < depth + 0.05))[0]
                # Calculate the mean WC for the current depth if indices are found
                if indices.size > 0:
                    mean_wc = np.mean(self.thetatop[timestep][indices])
                else:
                    mean_wc = np.nan  # Assign NaN if no indices are found in this depth range
                # Append the mean WC for the current depth
                self.mean_wc_dict[f"{depth}m"].append(mean_wc)
        return self.mean_wc_dict