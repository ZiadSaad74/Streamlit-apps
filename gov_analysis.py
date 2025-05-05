import pandas as pd
import streamlit as st
import plotly.express as px
from datetime import timedelta

st.set_page_config(page_title="Leads Analysis by Governorates", layout="wide")

def load_data():
    df = pd.read_csv("locations.csv", na_values=["null"])  # Load new dataset
    df['Created On'] = pd.to_datetime(df['Created On'])
    df = df.fillna("")
    return df

df = load_data()

st.sidebar.title("Leads Analysis by Governorates 🌍")
st.sidebar.header("⚙️ Filter")

max_date = df['Created On'].max()
min_date = df['Created On'].min()

default_start_date = max(min_date, max_date - timedelta(days=365))
start_date = st.sidebar.date_input("Start Date", default_start_date, min_value=min_date, max_value=max_date)
end_date = st.sidebar.date_input("End Date", max_date, min_value=min_date, max_value=max_date)

counts = ['Interested','Interested but not available','Interested Follow Up Required','Booked']
df = df[df['Last Reached Call Result'].isin(counts)]

counts.sort()
counts.insert(0,'All results')
result_selection = st.sidebar.selectbox("Select result", counts )

locs = list(df['location'].unique())
locs.sort()
locs.insert(0,'All locations')
locations = st.sidebar.selectbox("Select location", locs )

agents = list(df.Owner.unique())
agents.sort()
agents.insert(0,"All agents")
agents_list  = st.sidebar.selectbox("Agent",agents)


df_filtered = df[(df['Created On'] >= pd.Timestamp(start_date)) & (df['Created On'] <= pd.Timestamp(end_date))]

if result_selection !='All results':
    df_filtered = df_filtered[df_filtered['Last Reached Call Result']==result_selection]

if agents_list !='All agents':
    df_filtered = df_filtered[df_filtered['Owner']==agents_list]
    
if locations !='All locations':
    df_filtered = df_filtered[df_filtered['location']==locations]


locations_counts = df_filtered.location.value_counts().reset_index()
locations_counts.columns = ['Location','Count']

result_counts = df_filtered['location'].value_counts().reset_index()
result_counts.columns = ['Location', 'Count']

result_counts2 = df_filtered['Last Reached Call Result'].value_counts().reset_index()
result_counts2.columns = ['Result', 'Count']

st.subheader(f'Governorates results analysis')
st.markdown(f'Result : <b>{result_selection}</b> <br> Agent : <b>{agents_list}</b><br> Total leads = <b>{len(df_filtered)}</b><br> Location : <b>{locations}</b>', unsafe_allow_html=True)

if not result_counts.empty:
    fig1 = px.bar(locations_counts, y='Location', x='Count', color='Location',orientation='h', title= "Distribution of most interested results VS location <br> [Interested - Interested but not available - Interested Follow Up Required - Booked]")
    fig2 = px.pie(locations_counts, names='Location', values='Count', title= "Perecentge of each location")
    fig3 = px.bar(result_counts2, x="Result", y= 'Count', title='Distribution of results')

    st.plotly_chart(fig1, use_container_width=True)
    st.plotly_chart(fig2, use_container_width=True)
    st.plotly_chart(fig3)

else:
    st.warning("No data available for the selected date range.")

with st.expander("Display the data 🗂️"):
    st.dataframe(df_filtered)