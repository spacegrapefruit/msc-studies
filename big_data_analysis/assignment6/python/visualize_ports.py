import logging
import folium
import pandas as pd
from folium.plugins import HeatMap
from scipy.spatial import ConvexHull


def create_convexhull_polygon(
    map_object,
    list_of_points,
    layer_name,
    line_color,
    fill_color,
    weight,
    text,
):
    """
    Create a convex hull polygon on a folium map from a list of points.
    """
    # Since it is pointless to draw a convex hull polygon around less than 3 points check len of input
    if len(list_of_points) < 3:
        return

    # Create the convex hull using scipy.spatial
    form = [list_of_points[i] for i in ConvexHull(list_of_points).vertices]

    # Create feature group, add the polygon and add the feature group to the map
    fg = folium.FeatureGroup(name=layer_name)
    fg.add_child(
        folium.vector_layers.Polygon(
            locations=form,
            color=line_color,
            fill_color=fill_color,
            weight=weight,
            popup=(folium.Popup(text)),
        )
    )
    map_object.add_child(fg)

    return map_object


def create_port_visualization_map(
    ports_df_pandas,
    filtered_points_df_pandas,
    output_filename="detected_ports_map.html",
):
    """
    Create a visualization map of detected ports and filtered AIS signals.
    """
    logging.info(f"Generating visualization map: {output_filename}")

    # determine map center
    map_center = [
        ports_df_pandas["center_lat"].mean(),
        ports_df_pandas["center_lon"].mean(),
    ]

    m = folium.Map(location=map_center, zoom_start=7, tiles="CartoDB positron")

    # show all filtered points as a heatmap background
    heat_data = [
        [row["Latitude"], row["Longitude"]]
        for index, row in filtered_points_df_pandas.iterrows()
    ]
    HeatMap(heat_data, radius=8, blur=8).add_to(
        folium.FeatureGroup(name="Vessel Signal Heatmap (Slow)").add_to(m)
    )

    # detected ports as markers
    ports_group = folium.FeatureGroup(name="Detected Ports").add_to(m)
    max_radius = 30
    min_radius = 5

    for idx, port in ports_df_pandas.iterrows():
        port_loc = [port["center_lat"], port["center_lon"]]
        current_score = port.get("norm_size_vessels", 0.0)
        if pd.isna(current_score):
            current_score = 0.0
        radius = min_radius + (max_radius - min_radius) * current_score

        popup_html = f"""
        <b>Port ID (DBSCAN Cluster):</b> {port["port_id"]}<br>
        <b>Center:</b> ({port["center_lat"]:.4f}, {port["center_lon"]:.4f})<br>
        <b>Unique Vessels:</b> {port["num_unique_vessels"]}<br>
        <b>AIS Signals in Cluster:</b> {port["num_signals"]}<br>
        <b>Size Score:</b> {current_score:.2f}
        """

        folium.CircleMarker(
            location=port_loc,
            radius=radius,
            popup=folium.Popup(popup_html, max_width=300),
            color="purple",
            fill=True,
            fill_color="mediumpurple",
            fill_opacity=0.25,
            tooltip=f"Port Cluster {int(port['port_id'])} (Vessels: {int(port['num_unique_vessels'])})",
        ).add_to(ports_group)

        # create convex hull polygons for each port
        create_convexhull_polygon(
            m,
            port["member_points"],
            layer_name=f"Port {port['port_id']} Convex Hull",
            line_color="cadetblue",
            fill_color="skyblue",
            weight=2,
            text=f"Port {port['port_id']} Convex Hull",
        )

        # add points for each port
        points_group = folium.FeatureGroup(
            name=f"Port {port['port_id']} Points"
        ).add_to(m)
        for p_point in port["member_points"]:
            folium.CircleMarker(
                [p_point["Latitude"], p_point["Longitude"]],
                radius=1,
                color="blue",
            ).add_to(points_group)

    folium.LayerControl().add_to(m)
    m.save(output_filename)
    logging.info(f"Map saved to {output_filename}")
