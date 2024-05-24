import streamlit as st
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
from io import BytesIO
import plotly.graph_objects as go


def plot_to_image():
    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    return buf

def main():
    st.title("Fuzzy Tipping System")

    st.write("""
    ## The Tipping Problem
    Let’s create a fuzzy control system which models how you might choose to tip at a restaurant.
    """)

    quality = ctrl.Antecedent(np.arange(0, 11, 1), 'quality')
    service = ctrl.Antecedent(np.arange(0, 11, 1), 'service')
    tip = ctrl.Consequent(np.arange(0, 26, 1), 'tip')

    quality.automf(3)
    service.automf(3)

   
    tip['low'] = fuzz.trimf(tip.universe, [0, 0, 13])
    tip['medium'] = fuzz.trimf(tip.universe, [0, 13, 25])
    tip['high'] = fuzz.trimf(tip.universe, [13, 25, 25])

    rule1 = ctrl.Rule(quality['poor'] | service['poor'], tip['low'])
    rule2 = ctrl.Rule(service['average'], tip['medium'])
    rule3 = ctrl.Rule(service['good'] | quality['good'], tip['high'])

    tipping_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])

    tipping = ctrl.ControlSystemSimulation(tipping_ctrl)

    service_rating = st.slider('Service Quality', 0.0, 10.0, 5.0, 0.1)
    food_rating = st.slider('Food Quality', 0.0, 10.0, 5.0, 0.1)

    tipping.input['quality'] = float(food_rating)
    tipping.input['service'] = float(service_rating)

    tipping.compute()

    st.write(f"Based on the provided ratings, the recommended tip percentage is: {tipping.output['tip']}%")

    st.subheader("Fuzzy Sets")
    quality_fig, ax = plt.subplots()
    tip_fig, ax = plt.subplots()
    tip.view(sim=tipping, ax=ax)
    tip_image = plot_to_image()
    st.image(tip_image, use_column_width=True, caption="Tip Percentage")
    
    quality['average'].view(ax=ax)
    quality_image = plot_to_image()
    st.image(quality_image, use_column_width=True, caption="Service Quality")

    quality_levels = np.arange(0, 11, 1)
    service_levels = np.arange(0, 11, 1)
    x, y = np.meshgrid(quality_levels, service_levels)
    z = np.zeros_like(x)

    for i in range(len(x)):
        for j in range(len(x[0])):
            tipping.input['quality'] = x[i, j]
            tipping.input['service'] = y[i, j]
            tipping.compute()
            z[i, j] = tipping.output['tip']

    fig = go.Figure(data=[go.Surface(x=x, y=y, z=z)])
    fig.update_layout(title='Tip Amount Based on Quality and Service', scene=dict(xaxis_title='Quality', yaxis_title='Service', zaxis_title='Tip'))
    st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
