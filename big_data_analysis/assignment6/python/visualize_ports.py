import logging
import folium
from folium.plugins import HeatMap
import pandas as pd


def create_port_visualization_map(
    ports_df_pandas,
    filtered_points_df_pandas,
    output_filename="detected_ports_map.html",
):
    logging.info(f"Generating visualization map (DBSCAN version): {output_filename}")

    # Determine map center
    if not ports_df_pandas.empty:
        map_center = [
            ports_df_pandas["center_lat"].mean(),
            ports_df_pandas["center_lon"].mean(),
        ]
    elif filtered_points_df_pandas is not None and not filtered_points_df_pandas.empty:
        map_center = [
            filtered_points_df_pandas["Latitude"].mean(),
            filtered_points_df_pandas["Longitude"].mean(),
        ]
    else:
        map_center = [56.0, 10.0]

    m = folium.Map(location=map_center, zoom_start=7, tiles="CartoDB positron")

    # Heatmap of slow vessel signals (remains the same)
    if filtered_points_df_pandas is not None and not filtered_points_df_pandas.empty:
        heat_data = [
            [row["Latitude"], row["Longitude"]]
            for index, row in filtered_points_df_pandas.iterrows()
        ]
        if heat_data:
            HeatMap(heat_data, radius=10, blur=10).add_to(
                folium.FeatureGroup(name="Vessel Signal Heatmap (Slow)").add_to(m)
            )

    # Detected Ports as markers
    if not ports_df_pandas.empty:
        ports_group = folium.FeatureGroup(name="Detected Ports (DBSCAN)").add_to(m)
        max_radius = 30
        min_radius = 5

        for idx, port in ports_df_pandas.iterrows():
            port_loc = [port["center_lat"], port["center_lon"]]
            current_score = port.get("composite_size_score", 0.0)
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
                color="purple",  # Changed color to distinguish from grid-based
                fill=True,
                fill_color="mediumpurple",
                fill_opacity=0.6,
                tooltip=f"Port Cluster {port['port_id']} (Vessels: {port['num_unique_vessels']})",
            ).add_to(ports_group)

            # To visualize DBSCAN cluster shapes, you'd typically plot the convex hull of points_in_cluster
            # or plot all individual points of the cluster. This is more advanced.
            # If 'member_points' were added to port_summary_list and ports_df_pandas:
            # if 'member_points' in port and port['member_points']:
            #     points_feature_group = folium.FeatureGroup(name=f"Port {port['port_id']} Points", show=False).add_to(ports_group)
            #     for p_point in port['member_points']:
            #         folium.CircleMarker([p_point['Latitude'], p_point['Longitude']], radius=1, color='red').add_to(points_feature_group)

    folium.LayerControl().add_to(m)
    m.save(output_filename)
    print(f"Map saved to {output_filename}")
