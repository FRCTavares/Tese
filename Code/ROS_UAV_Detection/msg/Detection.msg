# Single object detection message
Header header
uint32 class_id
string class_name
float32 confidence
geometry_msgs/Point center
geometry_msgs/Point bbox_min  # Top-left corner
geometry_msgs/Point bbox_max  # Bottom-right corner
float32 area
