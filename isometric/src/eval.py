import ezdxf
import math

# Load a DXF file
def load_dxf(file_path):
    return ezdxf.readfile(file_path)

# Calculate the distance between two points
def calculate_distance(point1, point2):
    return math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)

# Calculate the angle between two lines
def calculate_angle(line1, line2):
    dx1, dy1 = line1[1][0] - line1[0][0], line1[1][1] - line1[0][1]
    dx2, dy2 = line2[1][0] - line2[0][0], line2[1][1] - line2[0][1]
    angle1 = math.atan2(dy1, dx1)
    angle2 = math.atan2(dy2, dx2)
    return math.degrees(angle1 - angle2)

# Check if the angle is within a tolerance range
def is_angle_within_tolerance(angle1, angle2, tolerance=1.0):
    return abs(angle1 - angle2) <= tolerance

# Main evaluation function
def evaluate_dxf(handmade_dxf, generated_dxf):
    handmade_doc = load_dxf(handmade_dxf)
    generated_doc = load_dxf(generated_dxf)
    
    # Set tolerances
    distance_tolerance = 1.0  # Distance tolerance (mm)
    angle_tolerance = 1.0     # Angle tolerance (degrees)

    evaluation_results = []
    distance_match_count = 0
    angle_match_count = 0
    total_lines = 0

    # Compare each LINE entity from both DXFs
    for line1, line2 in zip(handmade_doc.modelspace().query('LINE'), generated_doc.modelspace().query('LINE')):
        # Evaluate distance between corresponding lines
        point1_handmade = (line1.dxf.start.x, line1.dxf.start.y)
        point2_handmade = (line1.dxf.end.x, line1.dxf.end.y)
        point1_generated = (line2.dxf.start.x, line2.dxf.start.y)
        point2_generated = (line2.dxf.end.x, line2.dxf.end.y)
        
        distance_handmade = calculate_distance(point1_handmade, point2_handmade)
        distance_generated = calculate_distance(point1_generated, point2_generated)

        distance_error = abs(distance_handmade - distance_generated)
        
        # Evaluate distance matching
        distance_match = distance_error <= distance_tolerance
        if distance_match:
            distance_match_count += 1

        # Calculate and evaluate angle between corresponding lines
        angle_handmade = math.degrees(math.atan2(point2_handmade[1] - point1_handmade[1], point2_handmade[0] - point1_handmade[0]))
        angle_generated = math.degrees(math.atan2(point2_generated[1] - point1_generated[1], point2_generated[0] - point1_generated[0]))

        angle_error = abs(angle_handmade - angle_generated)
        
        # Evaluate angle matching
        angle_match = is_angle_within_tolerance(angle_handmade, angle_generated, angle_tolerance)
        if angle_match:
            angle_match_count += 1

        # Append evaluation results
        total_lines += 1
        evaluation_results.append({
            'line_handmade': (point1_handmade, point2_handmade),
            'line_generated': (point1_generated, point2_generated),
            'distance_error': distance_error,
            'distance_match': distance_match,
            'angle_error': angle_error,
            'angle_match': angle_match
        })
    
    # Calculate matching percentages
    distance_match_percentage = (distance_match_count / total_lines) * 100 if total_lines > 0 else 0
    angle_match_percentage = (angle_match_count / total_lines) * 100 if total_lines > 0 else 0

    # Calculate overall matching percentage (average of distance and angle)
    overall_match_percentage = (distance_match_percentage + angle_match_percentage) / 2

    return evaluation_results, distance_match_percentage, angle_match_percentage, overall_match_percentage

# Display results
def display_results(results, distance_match_percentage, angle_match_percentage, overall_match_percentage):
    for result in results:
        print(f"Manual Drawing: {result['line_handmade']}")
        print(f"Generated Drawing: {result['line_generated']}")
        print(f"Distance Error: {result['distance_error']:.2f} mm")
        print(f"Distance Match: {'Match' if result['distance_match'] else 'No Match'}")
        print(f"Angle Error: {result['angle_error']:.2f} degrees")
        print(f"Angle Match: {'Match' if result['angle_match'] else 'No Match'}")
        print('-' * 50)

    print(f"Distance Match Percentage: {distance_match_percentage:.2f}%")
    print(f"Angle Match Percentage: {angle_match_percentage:.2f}%")
    print(f"Overall Match Percentage: {overall_match_percentage:.2f}%")

# Main execution
handmade_dxf = "data/outputs/isometric/0/pipe.dxf"  # Path to the manually created DXF file
generated_dxf = "data/outputs/isometric/0/pipe.dxf"  # Path to the automatically generated DXF file

evaluation_results, distance_match_percentage, angle_match_percentage, overall_match_percentage = evaluate_dxf(handmade_dxf, generated_dxf)
display_results(evaluation_results, distance_match_percentage, angle_match_percentage, overall_match_percentage)
