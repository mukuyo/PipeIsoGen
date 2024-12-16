import argparse
import ezdxf
import math

def load_dxf(file_path):
    return ezdxf.readfile(file_path)

def calculate_inverse_distance(point1, point2):
    distance = math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)
    return distance

def calculate_both_distances(point1, generate):
    distance_1 = abs(generate.dxf.start.x - point1[0])
    distance_2 = abs(generate.dxf.end.x - point1[0])
    distance1 = min(distance_1, distance_2)

    distance_3 = abs(generate.dxf.start.y - point1[1])
    distance_4 = abs(generate.dxf.end.y - point1[1])
    distance2 = min(distance_3, distance_4)

    distance = distance1 + distance2
    return distance

def calculate_inverse_angle(line1, line2):
    dx1, dy1 = line1[1][0] - line1[0][0], line1[1][1] - line1[0][1]
    dx2, dy2 = line2[1][0] - line2[0][0], line2[1][1] - line2[0][1]
    angle1 = math.atan2(dy1, dx1)
    angle2 = math.atan2(dy2, dx2)
    return math.degrees(angle2 - angle1)

def is_angle_within_tolerance(angle1, angle2, tolerance=1.0):
    angle = abs(angle1 - angle2)
    if angle > 179:
        angle = 0
    return angle <= tolerance

def evaluate_dxf(handmade_dxf, generated_dxf):
    handmade_doc = load_dxf(handmade_dxf)
    generated_doc = load_dxf(generated_dxf)

    distance_tolerance = 5.0
    angle_tolerance = 1.0

    evaluation_results = []
    matching_count = 0
    distance_match_count = 0
    angle_match_count = 0
    total_lines = 0
    distance_error_total = 0

    handmade_lines = list(handmade_doc.modelspace().query('LINE'))
    generated_lines = list(generated_doc.modelspace().query('LINE'))

    used_handmade_lines = set()

    for generate in generated_lines:
        point1_generated = (generate.dxf.start.x, generate.dxf.start.y)
        point2_generated = (generate.dxf.end.x, generate.dxf.end.y)

        min_line = None
        min_distance = float('inf')

        for i, handmade in enumerate(handmade_lines):
            if i in used_handmade_lines:
                continue

            point1_handmade = (handmade.dxf.start.x, handmade.dxf.start.y)
            point2_handmade = (handmade.dxf.end.x, handmade.dxf.end.y)

            start_distance1 = calculate_both_distances(point1_handmade, generate)
            end_distance1 = calculate_both_distances(point2_handmade, generate)
            start_distance2 = calculate_both_distances(point1_generated, handmade)
            end_distance2 = calculate_both_distances(point2_generated, handmade)

            total_distance = start_distance1 + end_distance1 + start_distance2 + end_distance2

            if total_distance < min_distance:
                min_distance = total_distance
                min_line = (i, handmade)

        if min_line is None:
            continue

        used_handmade_lines.add(min_line[0])
        handmade = min_line[1]

        point1_handmade = (handmade.dxf.start.x, handmade.dxf.start.y)
        point2_handmade = (handmade.dxf.end.x, handmade.dxf.end.y)

        distance_handmade = calculate_inverse_distance(point1_handmade, point2_handmade)
        distance_generated = calculate_inverse_distance(point1_generated, point2_generated)
        distance_error = abs(distance_handmade - distance_generated)

        angle_handmade = math.degrees(math.atan2(point2_handmade[1] - point1_handmade[1], point2_handmade[0] - point1_handmade[0]))
        angle_generated = math.degrees(math.atan2(point2_generated[1] - point1_generated[1], point2_generated[0] - point1_generated[0]))

        distance_match = distance_error <= distance_tolerance
        angle_match = is_angle_within_tolerance(angle_handmade, angle_generated, angle_tolerance)

        distance_match_count += distance_match
        angle_match_count += angle_match

        if distance_match and angle_match:
            matching_count += 1

        distance_error_total += distance_error

        total_lines += 1
        evaluation_results.append({
            'line_handmade': (point1_handmade, point2_handmade),
            'line_generated': (point1_generated, point2_generated),
            'distance_error': distance_error,
            'distance_match': distance_match,
            'angle_error': abs(angle_generated - angle_handmade),
            'angle_match': angle_match
        })

    matching_percentage = (matching_count / total_lines) * 100 if total_lines > 0 else 0
    distance_match_percentage = (distance_match_count / total_lines) * 100 if total_lines > 0 else 0
    angle_match_percentage = (angle_match_count / total_lines) * 100 if total_lines > 0 else 0
    distance_error_avg = distance_error_total / total_lines if total_lines > 0 else 0

    return evaluation_results, matching_percentage, distance_match_percentage, angle_match_percentage, distance_error_total, distance_error_avg

def display_results(results, matching_percentage, distance_match_percentage, angle_match_percentage, distance_error_total, distance_error_avg):
    for result in results:
        print(f"Manual Drawing: {result['line_handmade']}")
        print(f"Generated Drawing: {result['line_generated']}")
        print(f"Distance Error: {result['distance_error']:.5f} mm")
        print(f"Distance Match: {'Match' if result['distance_match'] else 'No Match'}")
        print(f"Angle Error: {result['angle_error']:.2f} degrees")
        print(f"Angle Match: {'Match' if result['angle_match'] else 'No Match'}")
        print('-' * 50)

    print(f"Distance Matching Percentage: {distance_match_percentage:.2f}%")
    print(f"Angle Matching Percentage: {angle_match_percentage:.2f}%")
    print(f"Matching Percentage: {matching_percentage:.2f}%")
    print(f"Total Distance Error: {distance_error_total:.2f} mm")
    print(f"Average Distance Error: {distance_error_avg:.2f} mm")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate DXF files for similarity.")
    parser.add_argument("--gt_dxf_path", required=True, help="Path to the handmade DXF file.")
    parser.add_argument("--pred_dxf_path", required=True, help="Path to the generated DXF file.")

    args = parser.parse_args()

    evaluation_results, matching_percentage, distance_match_percentage, angle_match_percentage, distance_error_total, distance_error_avg = evaluate_dxf(args.gt_dxf_path, args.pred_dxf_path)
    display_results(evaluation_results, matching_percentage, distance_match_percentage, angle_match_percentage, distance_error_total, distance_error_avg)
