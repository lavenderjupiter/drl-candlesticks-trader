#!/usr/bin/env python
# coding: utf-8

import random
from gym import make
import numpy as np
from PIL import Image,ImageDraw

from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

from libs.utilities import get_funding_rate

scaler = MinMaxScaler()


TIME_UNIT = {'1m':60000, '3m':180000, '5m':300000, '15m':900000, '30m':1800000, '1h':3600000, '2h':7200000, 
             '4h':14400000, '6h':21600000, '8h':28800000, '12h':43200000, '1d':86400000, '3d':259200000, '1w':604800000} # milliseconds

def draw_candel(image, ohlc_price, location_xy, ratio, colors, width=6, wick_width=2):
    """    
    ohlc_price = [open_price, high_price, low_price, close_price]    
    location_xy = [x, y], left upper corner of the area for candelstick chart (in pixel) 
    ratio = pixel / price
    colors = [up, down] set the color for price up and down (default green and red)   
    width: width of candel stick (in pixel)
    wick_width: width of candel wick (in pixel)
    """
    # get the ohlc price
    open_price, high_price, low_price, close_price = [price for price in list(ohlc_price)]
    # get the x, y pixel coordinate of a canle (left upper corner)
    x, y = location_xy[0], location_xy[1]   
    # set the display color for price trend
    color_up, color_down, color_stay = [color for color in colors]
    
    draw = ImageDraw.Draw(image)
    color = color_up if close_price > open_price else color_down if close_price < open_price else color_stay
    height = (high_price - low_price) * ratio
    
    #draw the wick high_low line
    wick_x = x + width / 2 
    draw.line([(wick_x, y),(wick_x, y + height)], fill=color, width=wick_width)
    
    #draw the candel open_close rectangle
    if open_price != close_price:
        open_y = y + (high_price - open_price) * ratio #int
        close_y = y + (high_price - close_price) * ratio #int
        y0, y1 = min(open_y, close_y), max([open_y, close_y]) # y2 should >= y0
        draw.rectangle([(x, y0),(x + width, y1)], fill=color)
    else:
        open_y = y + (high_price - open_price) * ratio #int
        draw.line([(x, open_y),(x + width, open_y)], fill=color, width=wick_width)
        
        
def draw_volume(image, volume, location_xy, ratio, color, width=6):
    """
    volumn: volumn of the current duration    
    location_xy = [x, y], left lower corner of the area for volumn chart (in pixel) 
    colors = [up, down] set the color for price up and down (default green and red)
    ratio = pixel / price
    """
    x, y = location_xy[0], location_xy[1]
    
    draw = ImageDraw.Draw(image)
    height = volume * ratio
    draw.rectangle([(x, y - height),(x + width, y)], fill=color)    

def draw_hline(image, position_xy, ratio, total_width, color=(125, 125, 215), height=2):
    x1, y1 = position_xy

    x2 = x1 + ratio * total_width
    y2 = y1 + height

    draw = ImageDraw.Draw(image)
    draw.rectangle([(x1, y1),(x2, y2)], fill=color)

def draw_vline(image, position_xy, ratio, total_height, color=(125, 125, 215), width=2):
    x1, y2 = position_xy

    x2 = x1 + width
    y1 = y2 - ratio * total_height # y2 >= y1

    draw = ImageDraw.Draw(image)
    draw.rectangle([(x1, y1),(x2, y2)], fill=color)


def draw_moving_average(image, mas, xes, y, ratio, color, line_width=2):
    """
    mas = [moving average for point1, moving average for point2]
    points_xy = [(point1_x, point1_y), (point2_x, point2_y)]
    ratio = pixel / price
    """    
    ma_point1, ma_point2 = mas[0], mas[1]
    x_point1, x_point2 = xes[0], xes[1]
    
    y_point1 = ma_point1 * ratio + y
    y_point2 = ma_point2 * ratio + y
    
    draw = ImageDraw.Draw(image)
    draw.line([(x_point1, y_point1),(x_point2, y_point2)], fill=color, width=line_width)


def get_min_time_unit(time_units):
    time_ms = [TIME_UNIT[tu] for tu in time_units]
    min_ms = min(time_ms)
    
    position = time_ms.index(min_ms)
    min_unit = time_units[position]
    
    return min_unit


class PlotChart:  
    def __init__(self, dataframes_half, dataframes_all, dataframes, fund_rate_df, min_unit, periods, columns, colors=None):
        self.dataframes_half = dataframes_half
        self.dataframes_all = dataframes_all
        self.dataframes = dataframes
        self.fundrate_df = fund_rate_df
        self.min_unit = min_unit
        self.periods = periods
        self.columns = columns
        self.TIME_UNIT = TIME_UNIT

        if colors is None: 
            self.color_up, self.color_down, self.color_stay = (0, 255, 0), (255, 0, 0), (255, 255, 255)
            self.color_ma = [127, 170, 212, 255]
        else:
            self.color_up, self.color_down, self.color_stay = colors['up'], colors['down'], colors['stay']  
            self.color_ma = [colors['MA1'], colors['MA2'], colors['MA3'], colors['MA4']]  


    def plot_candel_image_df3(self, image, current_index, window_size=24, space=3, width=6, plot_ma=True, plot_gauge=False):
        """
        dataframes = {time_unit:dataframe, ...}
        window_size: number of candels in the chart
        current_index: current index of the min time unit
        min_unit: min time unit among the dataframes
        periods: simple moving average length
        colors={'up':(R,G,B), 'down':(R,G,B), 'stay':(R,G,B), 'MA1':B, 'MA2':B, 'MA3':B}
        """
        h, w = image.size #224x224
        blue_channel = Image.new('L', (h, w)) # create a gray image for drawing moving average as blue channle
        min_time_unit_df = self.dataframes_all[self.min_unit]
        current_ts = int(min_time_unit_df.iloc[current_index].Timestamp)    
        current_dt = min_time_unit_df.iloc[current_index].name
        fund_rate = get_funding_rate(self.fundrate_df, current_dt)
                                                                
        unit_row_height = 44
        space_h, space_v, space_lr = 3, 1, 2
        half_window_size = int(window_size / 2)
        
        # row #1, plot two charts in one row
        current_x = 0
        row_height = unit_row_height * 2
        for time_unit, df in self.dataframes_half.items():
            start_index = current_index + 1
            if TIME_UNIT[time_unit] > TIME_UNIT[self.min_unit]:
                start_index = np.max(np.where(df['Timestamp'].le(current_ts)))
                
            window_df = df.iloc[start_index-half_window_size:start_index, :]
            chart_df = window_df.loc[:, self.columns]
            chart_min = chart_df.stack().min()
            chart_max = chart_df.stack().max()                       
            ratio_chart = row_height / (chart_max - chart_min)   
            
            for i in range(half_window_size):
                ohlc_price = chart_df.iloc[i, :4]

                offset_x = i * (width + space_h) + space_lr + current_x            
                offset_y = (chart_max - ohlc_price['High']) * ratio_chart

                chart_xy = [offset_x, space_v + offset_y]
                draw_candel(image, ohlc_price, chart_xy, ratio_chart, [self.color_up, self.color_down, self.color_stay], width=width)
                
                mas = []
                if plot_ma:
                    if i == 0:
                        xes = [offset_x + width / 2, offset_x + width / 2]
                        for shift in range(len(self.periods)):
                            mas.append([chart_max - chart_df.iloc[0, 4 + shift], 
                                        chart_max - chart_df.iloc[0, 4 + shift]]) # y is calculated from top to bottom
                    else:
                        xes = [offset_x - space_h - width / 2, offset_x + width / 2] 
                        for shift in range(len(self.periods)):
                            mas.append([chart_max - chart_df.iloc[i - 1, 4 + shift], 
                                        chart_max - chart_df.iloc[i, 4 + shift]]) # y is calculated from top to bottom    

                    idx = len(self.color_ma) - len(self.periods)
                    for i, ma in enumerate(mas):
                        draw_moving_average(blue_channel, ma, xes, space_v, ratio_chart, self.color_ma[i + idx]) # color (255, 192, 0)            

            current_x = offset_x + width + space_h + space_lr * 2  
        
        # row #2, plot one chart in one row     
        row_height = unit_row_height * 3
        time_unit, min_df = list(self.dataframes_all.items())[0]
        start_index = current_index + 1
        window_df = min_df.iloc[(start_index - window_size):start_index, :]

        chart_df = window_df.loc[:, self.columns]
        chart_min = chart_df.stack().min() # min value in current window
        chart_max = chart_df.stack().max() # max value in current window                      
        ratio_chart = row_height / (chart_max - chart_min)        
        
        for i in range(window_size):
            ohlc_price = chart_df.iloc[i, :4]

            offset_x = i * (width + space_h) + 4            
            offset_y = (chart_max - ohlc_price['High']) * ratio_chart + space_v

            chart_xy = [offset_x, (space_v + unit_row_height * 2) + space_v * 2 + offset_y]
            draw_candel(image, ohlc_price, chart_xy, ratio_chart, [self.color_up, self.color_down, self.color_stay], width=width)

            mas = []
            if plot_ma:
                if i == 0:
                    xes = [offset_x + width / 2, offset_x + width / 2]
                    for shift in  range(len(self.periods)):
                        mas.append([chart_max - chart_df.iloc[0, 4 + shift], 
                                    chart_max - chart_df.iloc[0, 4 + shift]]) # y is calculated from top to bottom
                else:
                    xes = [offset_x - space_h - width / 2, offset_x + width / 2]     
                    for shift in range(len(self.periods)):
                        mas.append([chart_max - chart_df.iloc[i - 1, 4 + shift], 
                                    chart_max - chart_df.iloc[i, 4 + shift]])    

                current_y = space_v + (unit_row_height * 2 + space_v * 2)
                idx = len(self.color_ma) - len(self.periods)
                for i, ma in enumerate(mas):
                    draw_moving_average(blue_channel, ma, xes, current_y, ratio_chart, self.color_ma[i + idx]) # (255, 192, 0)                
            
            
        # make a trend label
        this_close, next_close = min_time_unit_df.iloc[current_index:(current_index + 2)]['Close'].values
        scaled_price = min_time_unit_df.iloc[current_index]['Scaled']
        price = '[{:.4f},{},{},{:.4f}]'.format(scaled_price, this_close, next_close, fund_rate)


        # rebuild the RBG image
        if plot_ma:
            image_RGB = np.array(image)
            image_B = np.array(blue_channel)
            image_RGB[:, :, 2] = image_B
            image = Image.fromarray(image_RGB.astype('uint8')).convert('RGB')

        if plot_gauge:
            draw_vline(image, [w / 2 - space_lr / 2, unit_row_height * 2 + space_v], scaled_price, row_height, color=(0, 0, 255))        

        return image, current_dt, price