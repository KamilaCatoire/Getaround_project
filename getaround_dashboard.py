import streamlit as st
from streamlit_lottie import st_lottie
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import numpy as np
import math

def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

def get_checkout_state(row):
    state = 'Unknown'
    if row['state'] == 'ended':
        if row['delay_at_checkout_in_minutes'] <= 0:
            state = "On time checkout"
        elif row['delay_at_checkout_in_minutes'] > 0:
            state = "Late checkout"
    if row['state'] == 'canceled':
        state = "Canceled"
    return state

def get_previous_rental_delay(row, dataframe):
    delay = np.nan
    if not math.isnan(row['previous_ended_rental_id']):
        delay = dataframe[dataframe['rental_id'] == row['previous_ended_rental_id']]['delay_at_checkout_in_minutes'].values[0]
    return delay

def get_impact_of_previous_rental_delay(row):
    impact = 'No previous rental filled out'
    if not math.isnan(row['checkin_delay_in_minutes']):
        if row['checkin_delay_in_minutes'] > 0:
            if row['state'] == 'Canceled':
                impact = 'Cancelation'
            else:
                impact = 'Late checkin'
        else:
            impact = 'No impact'
    return impact

def keep_only_ended_rentals(dataframe):
    return dataframe[(dataframe['state'] == 'On time checkout') | (dataframe['state'] == 'Late checkout')]

def keep_only_late_checkins_canceled(dataframe):
    return dataframe[(dataframe['checkin_delay_in_minutes'] > 0) & (dataframe['state'] == 'Canceled')]

def apply_threshold(dataframe, threshold, scope):
    if scope == 'All':
        rows_to_drop_df = dataframe[dataframe['time_delta_with_previous_rental_in_minutes'] < threshold]
    elif scope == "'Connect'":
        rows_to_drop_df = dataframe[(dataframe['time_delta_with_previous_rental_in_minutes'] < threshold) & (dataframe['checkin_type'] == 'connect')]
    nb_ended_rentals_dropped = len(keep_only_ended_rentals(rows_to_drop_df))
    nb_late_checkins_cancelations_dropped = len(keep_only_late_checkins_canceled(rows_to_drop_df))
    output = (
        dataframe.drop(rows_to_drop_df.index),
        nb_ended_rentals_dropped,
        nb_late_checkins_cancelations_dropped  
    )
    return output

def detect_outliers(dataframe, feature_name):
    q1 = dataframe[feature_name].quantile(0.25)
    q3 = dataframe[feature_name].quantile(0.75)
    interquartile_range = q3 - q1
    upper_fence = math.ceil(q3 + 1.5 * interquartile_range)
    nb_rows = len(dataframe)
    mask = (dataframe[feature_name] <= upper_fence) | (dataframe[feature_name].isna())
    nb_rows_without_outliers = len(dataframe[mask])
    nb_outliers = nb_rows - nb_rows_without_outliers
    percent_outliers = round(nb_outliers / nb_rows * 100)
    output = {
        'upper_fence': upper_fence,
        'percent_outliers': percent_outliers
    }
    return output

def vertical_space(height):
    for i in range(height):
        st.text('')


###########################
# MAIN CODE ###############
###########################

###################################################################################################
## PAGE and GRAPHICS CONFIGURATION
###################################################################################################

### Page style
st.set_page_config(
    page_title="GetAround Late Checkouts Analysis",
    page_icon="🚘⏱",
    layout="wide"
)
### Configure plotly display
config = {'displayModeBar': False} # Hide plotly modebar in figures

### Configure default plotly template
pio.templates['my_template'] = go.layout.Template(
    layout_margin_t = 50,
    layout_margin_b = 50
    )
pio.templates.default = 'my_template'

### Load gifs
right_arrow_gif = load_lottieurl('https://assets10.lottiefiles.com/packages/lf20_qrmoBnUE5w.json')

### Set graphic chart
colors = {
    'OK':'darkgreen',
    'Medium':'mediumspringgreen',
    'Bad':'lightgreen',
    'Other':'palegreen',
    'Unknown':'seagreen'}
###################################################################################################
## HEADER
###################################################################################################

### Title and description
st.title('🚘 GetAround Late Checkouts Analysis ⏱')
st.markdown("Sometimes, there are instances where users who have rented a car on GetAround arrive late for their check-out, which can disrupt the subsequent rental of the same vehicle and negatively affect the overall service quality and customer satisfaction. 🚘⏱  \n" \
    "To address this issue, one possible solution is to introduce a minimum time gap between two rentals. In other words, if the requested check-in or check-out times are too close to an already booked rental, the car would not be displayed in the search results. \n" \
    "However, implementing such a minimum time gap would have financial implications for both GetAround and car owners. Therefore, the purpose of this analysis is to provide insights that can aid in the decision-making process regarding the following factors: \n" \
    "- The **threshold** (the minimum time gap required between two rentals) \n" \
    "- The **scope** of the threshold's application (whether it should be applied to all cars or only to 'Connect'* cars).")
st.caption("_\* 'Connect cars': the driver doesn’t meet the owner and opens the car with his smartphone_")

### Tabs
analysis_tab, simulations_tab = st.tabs(["Analysis", "Simulations"])

###################################################################################################
## DATASET PROCESSINGS
###################################################################################################

### Load raw data
DATA_PATH = "get_around_delay_analysis.xlsx"
@st.cache_data
def load_data(nrows):
    data = pd.read_excel(DATA_PATH, nrows=nrows)
    return data
raw_df = load_data(None)

### Processing data
df = raw_df.copy()
#### modify 'state' column to add information on whether checkout is on time or late:
df['state'] = df.apply(get_checkout_state, axis = 1)
#### liken early checkouts (negative delay) to 'on-time' (zero delay) in 'delay_at_checkout_in_minutes' column:
df['delay_at_checkout_in_minutes'] = df['delay_at_checkout_in_minutes'].apply(lambda x: 0 if x < 0 else x)
#### add 'previous_rental_checkout_delay_in_minutes' column:
df['previous_rental_checkout_delay_in_minutes'] = df.apply(get_previous_rental_delay, args = [df], axis = 1)
#### add 'checkin_delay_in_minutes' column:
df['checkin_delay_in_minutes'] = df['previous_rental_checkout_delay_in_minutes'] - df['time_delta_with_previous_rental_in_minutes']
df['checkin_delay_in_minutes'] = df['checkin_delay_in_minutes'].apply(lambda x: 0 if x < 0 else x)
#### add 'impact_of_previous_rental_delay' column:
df['impact_of_previous_rental_delay'] = df.apply(get_impact_of_previous_rental_delay, axis = 1)
#### order by rental id:
df = df.sort_values('rental_id')

### Filtered datasets
late_checkouts_df = df[df['state'] == 'Late checkout']
previous_rental_delay_df = df[df['previous_rental_checkout_delay_in_minutes'] > 0]
late_checkins_df = df[df['checkin_delay_in_minutes'] > 0]
late_checkins_canceled_df = keep_only_late_checkins_canceled(df)

### Useful statistics
nb_rentals = len(df)
nb_ended_rentals = len(keep_only_ended_rentals(df))
nb_canceled_rentals = len(df[df['state'] == 'Canceled'])
nb_late_checkouts = len(late_checkouts_df)
late_checkouts_upper_fence = detect_outliers(late_checkouts_df, 'delay_at_checkout_in_minutes')['upper_fence']
nb_late_checkins = len(late_checkins_df)
late_checkins_upper_fence = detect_outliers(late_checkins_df, 'checkin_delay_in_minutes')['upper_fence']
late_checkins_canceled_upper_fence = detect_outliers(late_checkins_canceled_df, 'checkin_delay_in_minutes')['upper_fence']
nb_late_checkins_cancelations = len(late_checkins_canceled_df)

###################################################################################################
## ANALYSIS TAB
###################################################################################################

with analysis_tab:
    
    ### Show data
    show_data = st.selectbox('Show data ?', ['hide', 'show raw data', 'show processed data'])
    if show_data == 'show raw data': 
            st.write(raw_df)
            st.write(f"{len(raw_df)} rows")
    if show_data == 'show processed data': 
            st.markdown(
                "_Processings made:_\n" \
                "- _modify 'state' column to add information on whether checkout is on time or late,_\n" \
                "- _liken early checkouts (negative delay) to 'on-time' (zero delay)_\n " \
                "- _add 'previous_rental_checkout_delay_in_minutes', 'checkin_delay_in_minutes' and 'impact_of_previous_rental_delay' columns_\n" \
                "- _sort by rental id_"
                )
            st.write(df)
            
            st.write(f"{len(df)} rows")
    st.markdown("_Note: all analyses below are made on processed data_")

    ### Main metrics
    st.header('Main metrics of dataset')
    main_metrics_cols = st.columns([20,30,50])
    with main_metrics_cols[0]:
        st.metric(label = "Number of rentals", value= nb_rentals)
        st.metric(label = "Number of cars", value= df['car_id'].nunique())
    with main_metrics_cols[2]:
        st.metric(label = "Share of 'Connect' rentals", value= f"{round(len(df[df['checkin_type'] == 'connect']) /nb_rentals * 100)}%")
        st.metric(label = "Share of consecutive rentals of a same car", value= f"{round(len(df[~df['previous_ended_rental_id'].isna()]) /nb_rentals * 100)}%")
    with main_metrics_cols[1]:
        st.metric(
            label = "Max. delta between consecutive rentals", 
            value= f"{round(previous_rental_delay_df['time_delta_with_previous_rental_in_minutes'].max())} minutes")
        st.metric(
            label = "Average delta between consecutive rentals", 
            value= f"{round(previous_rental_delay_df['time_delta_with_previous_rental_in_minutes'].mean())} minutes")
    
       ### Checkouts overview
    st.header('Checkouts overview')
    checkouts_info_cols = st.columns([35, 40, 25])
    with checkouts_info_cols[0]:
        state_pie = px.pie(
            df, names = "state", color = "state", 
            height = 500, 
            color_discrete_map={
                'On time checkout':colors['OK'], 
                'Late checkout':colors['Bad'], 
                'Canceled':colors['Other'],
                'Unknown':colors['Unknown']
                },
            category_orders={"state": ['On time checkout', 'Late checkout', 'Canceled', 'Unknown']},
            title = "<b>Rental state</b>",
            hole=0.3)
        
        st.plotly_chart(state_pie, use_container_width=True, config = config)
        
    histogram = px.histogram(
            previous_rental_delay_df,
            x = 'time_delta_with_previous_rental_in_minutes',
            color='checkin_type',
            # title = 'Histogram of delta between consecutive rentals',
            # labels={'time_delta_with_previous_rental_in_minutes':'time delta with previous rental in minutes'},
            nbins=100, 
            histfunc= 'sum',
            marginal='box',
            text_auto=True, 
            barmode='overlay',
            color_discrete_map={"mobile": "palegreen", "connect": "darkgreen"},
        )

    histogram.update_layout(
        title_text='Histogram of time delta between consecutive rentals', # title of plot
        xaxis_title_text='Time delta with previous rental in minutes', # xaxis label
        yaxis_title_text='Count', # yaxis label
    )
    st.plotly_chart(histogram, use_container_width=True, config = config)

    with checkouts_info_cols[1]:
        delays_boxplot = px.box(
            late_checkouts_df, y = 'delay_at_checkout_in_minutes', height = 500,
            labels = {'delay_at_checkout_in_minutes': 'Checkout delay (minutes)'},
            range_y = [0, late_checkouts_upper_fence + 1],
            title = "<b>Checkout delays breakdown (outliers hidden)</b>")
        delays_boxplot.update_traces(marker_color = colors['Bad'])
        st.plotly_chart(delays_boxplot, use_container_width=True, config = config)
    with checkouts_info_cols[2]:
        vertical_space(4)
        st.metric(
            label = "", 
            value = f"{round(detect_outliers(df, 'delay_at_checkout_in_minutes')['percent_outliers'])}%",
            delta = f"checkout delays are outliers (> {late_checkouts_upper_fence} minutes)",
            delta_color = 'normal'
            )
        st.metric(
            label = "", 
            value =f"{round(len(late_checkouts_df[late_checkouts_df['delay_at_checkout_in_minutes'] >= 60]) / nb_late_checkouts * 100)}% of late checkouts", 
            delta = "have a delay of at least 1 hour",
            delta_color = 'normal'
            )
        st.write()
        vertical_space(1)
        st.metric(
            label = "", 
            value =f"{round(len(late_checkouts_df[late_checkouts_df['delay_at_checkout_in_minutes'] <= 30]) / nb_late_checkouts * 100)}% of late checkouts",
            delta = "have a delay of less than 30 minutes",
            delta_color = 'normal'
        )

    ### Impacts on checkins
    st.header('Impacts of delays on next checkin (when existing)')
    checkins_info_cols = st.columns([35, 40, 25])
    with checkins_info_cols[0]:
        impacts_pie = px.pie(
            previous_rental_delay_df, names = "impact_of_previous_rental_delay", color = "impact_of_previous_rental_delay", 
            height = 500, 
            color_discrete_map={
                'No impact':colors['OK'], 
                'Late checkin':colors['Medium'], 
                'Cancelation':colors['Bad'],
                'No previous rental filled out':colors['Unknown']
                },
            category_orders={"impact_of_previous_rental_delay": ['No impact', 'Late checkin', 'Cancelation', 'No previous rental filled out']},
            title = "<b>Impacts of checkout delays on checkin state</b>",
            hole=0.3)
        st.plotly_chart(impacts_pie, use_container_width=True, config = config)
    with checkins_info_cols[1]:
        checkin_delays_boxplot = go.Figure()
        checkin_delays_boxplot.add_trace(
            go.Box(
                y = late_checkins_df['checkin_delay_in_minutes'], 
                name = 'Late checkin', marker_color = colors['Medium']
            )
        )
        checkin_delays_boxplot.add_trace(
            go.Box(
                y = late_checkins_canceled_df['checkin_delay_in_minutes'], 
                name = 'Canceled', marker_color = colors['Bad']
            )
        )
        checkin_delays_boxplot.update_layout(
            showlegend = False, boxgap = 0.25, 
            yaxis_range = [0, late_checkins_canceled_upper_fence + 1],
            title = "<b>Checkin delays breakdown (some outliers hidden)</b>"
        )
        st.plotly_chart(checkin_delays_boxplot, use_container_width=True, config = config)
    with checkins_info_cols[2]:
        vertical_space(3)
        st.metric(
            label = "", 
            value=f"{round(nb_late_checkins / nb_rentals * 100)}% of checkins",
            delta = "are late because of a previous checkout delay",
            delta_color = 'normal'
        )
        st.metric(
            label = "", 
            value=f"{round(nb_late_checkins_cancelations / nb_late_checkins * 100)}% of late checkins", 
            delta = "are cancelled",
            delta_color = 'normal'
            )
        st.write()
        vertical_space(1)
        st.metric(
            label = "", 
            value=f"{round(nb_late_checkins_cancelations / nb_canceled_rentals * 100)}% of all cancelations",
            delta = "follow a late checkin",
            delta_color = 'normal'
        )

###################################################################################################

###################################################################################################
# SIMULATIONS TAB
###################################################################################################

with simulations_tab:

    ### Input form
    st.markdown('**You can visualize here the impacts of applying a minimum delay between consecutive rentals:**')
    with st.form(key='simulation_form'):
        simulation_form_cols = st.columns([15, 15, 15, 12, 43])
        with simulation_form_cols[0]:
            simulation_threshold = st.number_input(label='Threshold', min_value = 15, step = 15)
        with simulation_form_cols[1]:
            simulation_scope = st.radio('Scope', ['All', "'Connect'"], key = 3)
        submit = st.form_submit_button(label='Run simulation 👈')

    ### Simulation results
    if submit:
        with_threshold_df, nb_ended_rentals_lost, nb_late_checkins_cancelations_avoided = apply_threshold(df, simulation_threshold, simulation_scope)
        previous_rental_delay_with_threshold_df = with_threshold_df[with_threshold_df['previous_rental_checkout_delay_in_minutes'] > 0]
        
        ##### Influence on business metrics
        with simulation_form_cols[2]:
            st_lottie(right_arrow_gif, height = 100)
        with simulation_form_cols[3]:
            st.metric(
                label = "", 
                value=f"{round(nb_ended_rentals_lost / nb_ended_rentals * 100, 1)}%", 
                delta = "revenue loss",
                delta_color = 'inverse'
                )
        with simulation_form_cols[4]:
            st.metric(
                label = "", 
                value=f"{round(nb_late_checkins_cancelations_avoided / nb_late_checkins_cancelations * 100)}%",
                delta = "late-checkins-related cancelations avoided",
                delta_color = 'normal'
            )
        
        #### Evolution of the impacts of checkout delays on checkin state
        st.markdown("**Rentals following a delayed one - evolution of impacts on checkins repartition:**")
        if len(previous_rental_delay_with_threshold_df['impact_of_previous_rental_delay']) == 0:
            late_checkouts_impact_evolution_cols = st.columns([30, 10, 5, 25, 30])
            with late_checkouts_impact_evolution_cols[3]:
                vertical_space(14)
                st.markdown("### _No more rentals consecutive to a delayed one_")
        else:
            late_checkouts_impact_evolution_cols = st.columns([30, 10, 30, 30])
            with late_checkouts_impact_evolution_cols[2]:
                impacts_pie_with_threshold = px.pie(
                    previous_rental_delay_with_threshold_df, 
                    names = "impact_of_previous_rental_delay", color = "impact_of_previous_rental_delay", 
                    height = 500, 
                    color_discrete_map={
                        'No impact':colors['OK'], 
                        'Late checkin':colors['Medium'], 
                        'Cancelation':colors['Bad'],
                        'No previous rental filled out':colors['Unknown']
                        },
                    category_orders={"impact_of_previous_rental_delay": ['No impact', 'Late checkin', 'Cancelation', 'No previous rental filled out']},
                    title = "<b>With threshold</b>",
                    hole=0.3)
                st.plotly_chart(impacts_pie_with_threshold, use_container_width=True, config = config)
        with late_checkouts_impact_evolution_cols[0]:
            impacts_pie_without_threshold = impacts_pie
            impacts_pie_without_threshold.update_layout(title = "<b>Without threshold</b>")
            st.plotly_chart(impacts_pie_without_threshold, use_container_width=True, config = config)
        with late_checkouts_impact_evolution_cols[1]:
            vertical_space(10)
            st_lottie(right_arrow_gif, height = 200)