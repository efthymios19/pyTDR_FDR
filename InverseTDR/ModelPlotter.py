import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np
from numpy import isnan
import math 
from datetime import datetime, timedelta

class TodoroffPlotter:
    def __init__(self, td):
        self.td = td
    
    def plot_reflcoefappdist(self, timesteps=None):
        """
        Plots the Reflection Coefficient vs Apparent Distance in m for specific timesteps.
        Parameters:
        timesteps (list of int): List of desired timesteps for plotting. If None, all timesteps will be plotted.
        """
        # Call the impulse_response method of the Todoroff instance
        timestamp, Vp, WaveAvg, points, cablelen, windowlen, probelen, measurements = self.td.import_csv()
        WaveformApparentDistance, WaveformResponseTime, ImpulseResponse, VDown = self.td.impulse_response()
        
        # Use all timesteps if none specified
        if timesteps is None:
            timesteps = range(len(timestamp))

        # Check timesteps are in valid range
        for timestep in timesteps:
            if timestep < 0 or timestep >= len(timestamp):
                raise ValueError(f"Timestep {timestep} is out of range. Valid range is 0 to {len(timestamp)-1}.")
            
        # Define subplot layout: 4 plots per row
        num_plots = len(timesteps)
        num_cols = 4
        num_rows = math.ceil(num_plots / num_cols)
        
        # Set up figure and subplots
        fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 4 * num_rows))
        axs = axs.flatten()  # Flatten to 1D array for easy indexing
        
        for i, timestep in enumerate(timesteps):
           # Check if timestep is valid
            if timestep >= len(timestamp):
                continue
            # Extract data for the specified timestep
            x_data = WaveformApparentDistance[timestep]
            y_data = measurements.iloc[timestep]
            
            # Plot data if x_data and y_data have the same length
            if len(x_data) == len(y_data):
                axs[i].plot(x_data, y_data, label=f'Date: {timestamp[timestep]}')
                axs[i].set_xlabel('Distance (m)')
                axs[i].set_ylabel('Reflection Coefficient')
                axs[i].set_title(f'Timestep {timestep}')
                axs[i].grid(True)
                axs[i].legend()
            
            # Remove any empty subplots
        for j in range(i + 1, len(axs)):
            fig.delaxes(axs[j])
        
        # Adjust layout and display plot
        plt.tight_layout()
        plt.show()

        

    def plot_reflcoeftime(self, timesteps=None):
        """
        Plots the Reflection Coefficient vs Time in m for specific timesteps.
        Parameters:
        timesteps (list of int): List of desired timesteps for plotting. If None, all timesteps will be plotted.
        """
        # Call the impulse_response method of the Todoroff instance
        timestamp, Vp, WaveAvg, points, cablelen, windowlen, probelen, measurements = self.td.import_csv()
        WaveformApparentDistance, WaveformResponseTime, ImpulseResponse, VDown = self.td.impulse_response()
        
        # Use all timesteps if none specified
        if timesteps is None:
            timesteps = range(len(timestamp))

        # Check timesteps are in valid range
        for timestep in timesteps:
            if timestep < 0 or timestep >= len(timestamp):
                raise ValueError(f"Timestep {timestep} is out of range. Valid range is 0 to {len(timestamp)-1}.")
            
        # Define subplot layout: 4 plots per row
        num_plots = len(timesteps)
        num_cols = 4
        num_rows = math.ceil(num_plots / num_cols)
        
        # Set up figure and subplots
        fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 4 * num_rows))
        axs = axs.flatten()  # Flatten to 1D array for easy indexing
        
        for i, timestep in enumerate(timesteps):
           # Check if timestep is valid
            if timestep >= len(timestamp):
                continue
            # Extract data for the specified timestep
            x_data = WaveformResponseTime[timestep]
            y_data = measurements.iloc[timestep]
            
            # Plot data if x_data and y_data have the same length
            if len(x_data) == len(y_data):
                axs[i].plot(x_data, y_data, label=f'Date: {timestamp[timestep]}')
                axs[i].set_xlabel('Time (ns)')
                axs[i].set_ylabel('Reflection Coefficient')
                axs[i].set_title(f'Timestep {timestep}')
                axs[i].grid(True)
                axs[i].legend()
            
            # Remove any empty subplots
        for j in range(i + 1, len(axs)):
            fig.delaxes(axs[j])
        
        # Adjust layout and display plot
        plt.tight_layout()
        plt.show()
    

    def plot_startendprobe(self, timesteps=None):
        """
        Plots Reflection Coefficient and 1st Derivative for each probe segment. Prints the 20 first indixes of probe segment on which the max values
        of 1st derivative are observed.

        Parameters:
        timesteps (list of int): List of desired timesteps for plotting. If None, all timesteps will be plotted
        """
        # Import required data
        datetime, Vp, WaveAvg, points, cablelen, windowlen, probelen, measurements = self.td.import_csv()
        Segments, dy_dt, dy_dt2, dy_dt_alt, indices = self.td.Selectstartendprobe()
        
        if timesteps is None:
            timesteps=range(points.shape[0])

        #Validate Timesteps
        for timestep in timesteps:
            if timestep<0 or timestep>=points.shape[0]:
                raise ValueError(f"Timestep {timestep} is out of range. Valid range is 0 to {points.shape[0] - 1}.")
        
        num_plots = len(timesteps)
        num_cols = 2  # Up to 2 plots per row
        num_rows = math.ceil(num_plots / num_cols)  # Total rows of subplots needed
        
        if num_plots == 1:
            fig, ax = plt.subplots(figsize=(10, 6))  # Smaller size for single plot
            axs = [ax]  # Wrap in a list for consistency in indexing
        else:
            fig, axs = plt.subplots(num_rows, num_cols, figsize=(10, 2 * num_rows))
            axs = axs.flatten()  # Flatten to 1D array for easy indexing
        
        for i, timestep in enumerate(timesteps):
            # Check if timestep is valid
            if timestep >= points.shape[0]:
                continue
            
            # Get current axis
            ax = axs[i]
            
            # Data selection for each probe
            num_points = points.loc[i]
            start_idx = max(0, indices[i][0] - 100)
            end_idx = indices[i][0] + int(num_points / 4)
            
            # Plot Reflection Coefficient
            ax.plot(Segments[timestep][start_idx:end_idx], measurements.iloc[timestep, start_idx:end_idx], color="black", label="Reflection Coefficient")
            
            # Plot 1st Derivative on the same axis
            ax.plot(Segments[timestep][start_idx:end_idx], dy_dt[timestep][start_idx:end_idx], color="purple", label="1st Derivative")
            # Set plot details
            ax.set_title(f'Timestep {timestep} - Reflection Coefficient and 1st Derivative')
            ax.set_xlabel('Segments (1 : n)')
            ax.set_ylabel('Value')
            ax.legend()
            ax.grid(True)
            
            # Display indices information on each plot
            indices_text = f"Indices {indices[i]}: Select start (1st) and end (2nd) of probe for water content computation."
            ax.annotate(indices_text, xy=(0.5, -0.25), xycoords='axes fraction', ha='center', fontsize=10, color="gray")
        # Remove any empty subplots
        for j in range(num_rows, len(axs)):
            fig.delaxes(axs[j])
        
        # Adjust layout and display plot
        plt.tight_layout()
        plt.show()  
    
    def plot_modelvsimpulse(self, timesteps=None):
        """
        Plots the Modeled Impulse from Todoroff model vs the Impulse response measured for specific timesteps.
        Parameters:
        timesteps (list of int): List of desired timesteps for plotting. If None, all timesteps will be plotted.
        """
        # Call the impulse_response method of the Todoroff instance
        timestamp, Vp, WaveAvg, points, cablelen, windowlen, probelen, measurements = self.td.import_csv()
        WaveformApparentDistance, WaveformResponseTime, ImpulseResponse, VDown= self.td.impulse_response()
        Segmentreflcoef, original_values, VDown, Vup=self.td.forwardmodel()
        
        # Use all timesteps if none specified
        if timesteps is None:
            timesteps = range(len(timestamp))

        # Check timesteps are in valid range
        for timestep in timesteps:
            if timestep < 0 or timestep >= len(timestamp):
                raise ValueError(f"Timestep {timestep} is out of range. Valid range is 0 to {len(timestamp)-1}.")
            
        # Define subplot layout: 4 plots per row
        num_plots = len(timesteps)
        num_cols = 4
        num_rows = math.ceil(num_plots / num_cols)
        # Set up figure and subplots
        if num_plots == 1:
            fig, ax = plt.subplots(figsize=(10, 6))  # Smaller size for single plot
            axs = [ax]  # Wrap in a list for consistency in indexing
        else:
            fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 4 * num_rows))
            axs = axs.flatten()  # Flatten to 1D array for easy indexing

        for i,timestep in enumerate(timesteps):
           # Check if timestep is valid
            if timestep >= len(timestamp):
                continue
            # Extract data for the specified timestep
            x_data = WaveformResponseTime[timestep]
            y_data = ImpulseResponse[timestep]
            model = Segmentreflcoef[timestep][0]
            
            # Plot data if x_data and y_data have the same length
            if len(x_data) == len(y_data):
                axs[i].plot(x_data, y_data, label=f'Impulse Response Date: {timestamp[timestep]}')
                axs[i].plot(x_data, model, label=f'Modeled Impulse Response Date: {timestamp[timestep]}')
                axs[i].set_xlabel('Time (ns)')
                axs[i].set_ylabel('Impulse Response')
                axs[i].set_title(f'Timestep {timestep}')
                axs[i].grid(True)
                axs[i].legend()
            
            # Remove any empty subplots
        for j in range(i + 1, len(axs)):
            fig.delaxes(axs[j])
        
        # Adjust layout and display plot
        plt.tight_layout()
        plt.show() 
    
    
    def plot_Impendance(self, timesteps, idxfirst, idxsec):
        """
        Plots Impendace vs Time (ns) for specific timesteps.
        Parameters:
        timesteps (list of int): List of desired timesteps for plotting. If None, all timesteps will be plotted.
        idxfirst (int): the segment of the first reflection at the start of the probe
        idxsec (int): the segment of the second reflection at the end of the probe
        """
        # Call the impulse_response method of the Todoroff instance
        timestamp, Vp, WaveAvg, points, cablelen, windowlen, probelen, measurements = self.td.import_csv()
        WaveformApparentDistance, WaveformResponseTime, ImpulseResponse, VDown= self.td.impulse_response()
        Impendance, Ka_all, lenact_all=self.td.inversemodel(idxfirst,idxsec)
        
        # Use all timesteps if none specified
        if timesteps is None:
            timesteps = range(len(timestamp))

        # Check timesteps are in valid range
        for timestep in timesteps:
            if timestep < 0 or timestep >= len(timestamp):
                raise ValueError(f"Timestep {timestep} is out of range. Valid range is 0 to {len(timestamp)-1}.")
            
        # Define subplot layout: 4 plots per row
        num_plots = len(timesteps)
        num_cols = 4
        num_rows = math.ceil(num_plots / num_cols)
        
        if num_plots == 1:
            fig, ax = plt.subplots(figsize=(10, 6))  # Smaller size for single plot
            axs = [ax]  # Wrap in a list for consistency in indexing
        else:
            fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 2 * num_rows))
            axs = axs.flatten()  # Flatten to 1D array for easy indexing

        for i, timestep in enumerate(timesteps):
           # Check if timestep is valid
            if timestep >= len(timestamp):
                continue
            # Extract data for the specified timestep
            x_data = WaveformResponseTime[timestep]
            y_data = Impendance[timestep][0]
            
            # Plot data if x_data and y_data have the same length
            if len(x_data) == len(y_data):
                axs[i].plot(x_data, y_data, label=f'Impendance Date: {timestamp[timestep]}')
                axs[i].set_xlabel('Time (ns)')
                axs[i].set_ylabel('Impendance (Ohms)')
                axs[i].set_title(f'Impendance {timestep}')
                axs[i].grid(True)
                axs[i].legend()
            
            # Remove any empty subplots
        for j in range(i + 1, len(axs)):
            fig.delaxes(axs[j])
        
        # Adjust layout and display plot
        plt.tight_layout()
        plt.show() 

    def plot_DielectricConstant(self, timesteps, idxfirst, idxsec):
        """
        Plots Dielectric constant for the whole length of the probe for specific timesteps.
        Parameters:
        timesteps (list of int): List of desired timesteps for plotting. If None, all timesteps will be plotted.
        idxfirst (int): the segment of the first reflection at the start of the probe
        idxsec (int): the segment of the second reflection at the end of the probe
        """
        # Call the impulse_response method of the Todoroff instance
        timestamp, Vp, WaveAvg, points, cablelen, windowlen, probelen, measurements = self.td.import_csv()
        Impendance, Ka, lenact=self.td.inversemodel(idxfirst,idxsec)
        
        # Use all timesteps if none specified
        if timesteps is None:
            timesteps = range(len(timestamp))

        # Check timesteps are in valid range
        for timestep in timesteps:
            if timestep < 0 or timestep >= len(timestamp):
                raise ValueError(f"Timestep {timestep} is out of range. Valid range is 0 to {len(timestamp)-1}.")
            
        # Define subplot layout: 4 plots per row
        num_plots = len(timesteps)
        num_cols = 4
        num_rows = math.ceil(num_plots / num_cols)
        
        if num_plots == 1:
            fig, ax = plt.subplots(figsize=(10, 6))  # Smaller size for single plot
            axs = [ax]  # Wrap in a list for consistency in indexing
        else:
            fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 2 * num_rows))
            axs = axs.flatten()  # Flatten to 1D array for easy indexing

        for i, timestep in enumerate(timesteps):
           # Check if timestep is valid
            if timestep >= len(timestamp):
                continue
            # Extract data for the specified timestep
            x_data = Ka[timestep]
            y_data = lenact[timestep]
            
            condition = lenact[timestep] > probelen[timestep]
            # Use np.where to get the indices where the condition is True
            indices = np.where(condition)[0]
            # Check if any indices were found
            if indices.size > 0:
                # Get the first index where lenact is greater than probelen
                first_index = indices[0]
                # Slice x_data and y_data up to the found index
                x_data = x_data[:first_index + 1]  # +1 to include the found index
                y_data = y_data[:first_index + 1]

                axs[i].plot(x_data, -y_data, label=f'Dielectric constant Date: {timestamp[timestep]}')
                axs[i].set_xlabel('Dielectric constant')
                axs[i].set_ylabel('Depth (m)')
                axs[i].set_title(f'Dielectric constant {timestep}')
                axs[i].grid(True)
                axs[i].legend()
            else:
                axs[i].plot(x_data, -y_data, label=f'Dielectric constant Date: {timestamp[timestep]}')
                axs[i].set_xlabel('Dielectric constant')
                axs[i].set_ylabel('Depth (m)')
                axs[i].set_title(f'Dielectric constant {timestep}')
                axs[i].grid(True)
                axs[i].legend()
            
            # Remove any empty subplots
        for j in range(i + 1, len(axs)):
            fig.delaxes(axs[j])
        
        # Adjust layout and display plot
        plt.tight_layout()
        plt.show() 

        
    def plot_WaterContent(self, timesteps=None):
        """
        Plots spatial water content for the whole length of the probe for specific timesteps.
        Parameters:
        timesteps (list of int): List of desired timesteps for plotting. If None, all timesteps will be plotted.
        If 'all', all timesteps will be plotted on one large plot. 
        """
        # Call the impulse_response method of the Todoroff instance
        timestamp, Vp, WaveAvg, points, cablelen, windowlen, probelen, measurements = self.td.import_csv()
        
        # Use all timesteps if none specified
        if timesteps is None:
            timesteps = range(len(timestamp))

        # Check timesteps are in valid range
        for timestep in timesteps:
            if timestep < 0 or timestep >= len(timestamp):
                raise ValueError(f"Timestep {timestep} is out of range. Valid range is 0 to {len(timestamp)-1}.")
            
        # Define subplot layout: 4 plots per row
        num_plots = len(timesteps)
        num_cols = 4
        num_rows = math.ceil(num_plots / num_cols)
        
        if num_plots == 1:
            fig, ax = plt.subplots(figsize=(10, 6))  # Smaller size for single plot
            axs = [ax]  # Wrap in a list for consistency in indexing
        else:
            fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 2 * num_rows))
            axs = axs.flatten()  # Flatten to 1D array for easy indexing

        for i, timestep in enumerate(timesteps):
           # Check if timestep is valid
            if timestep >= len(timestamp):
                continue
            # Extract data for the specified timestep
            x_data = self.td.thetatop[timestep]
            y_data = self.td.lenact_all[timestep]
            
            condition = self.td.lenact_all[timestep] > probelen[timestep]
            # Use np.where to get the indices where the condition is True
            indices = np.where(condition)[0]
            # Check if any indices were found
            if indices.size > 0:
                # Get the first index where lenact is greater than probelen
                first_index = indices[0]
                # Slice x_data and y_data up to the found index
                x_data = x_data[:first_index + 1]  # +1 to include the found index
                y_data = y_data[:first_index + 1]
            
            condition_ = self.td.thetatop[timestep] <=100
            # Use np.where to get the indices where the condition is True
            indices = np.where(condition_)[0]
            # Check if any indices were found
            if indices.size > 0:
                # Get the first index where lenact is greater than probelen
                last_index = indices[-1]
                # Slice x_data and y_data up to the found index
                x_data = x_data[:last_index + 1]  # +1 to include the found index
                y_data = y_data[:last_index + 1]
                axs[i].plot(x_data, -y_data, label=f'Date: {timestamp[timestep]}')
                axs[i].set_xlabel('Soil volumetric water content (vol.%)')
                axs[i].set_ylabel('Depth (m)')
                axs[i].set_title(f'Soil volumetric water content {timestep}')
                axs[i].grid(True)
                axs[i].legend()
            else:
                axs[i].plot(x_data, -y_data, label=f'Date: {timestamp[timestep]}')
                axs[i].set_xlabel('Soil volumetric water content (vol.%)')
                axs[i].set_ylabel('Depth (m)')
                axs[i].set_title(f'Soil volumetric water content {timestep}')
                axs[i].grid(True)
                axs[i].legend()
            
            # Remove any empty subplots
        for j in range(i + 1, len(axs)):
            fig.delaxes(axs[j])
        
        # Adjust layout and display plot
        plt.tight_layout()
        plt.show() 