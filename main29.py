import cv2
from ultralytics import YOLO
import pandas as pd
import numpy as np
from collections import defaultdict, OrderedDict

# Initialize YOLO model
model = YOLO('best2.pt')
cap = cv2.VideoCapture('input3.mp4')
class_list = ["Counter", "Customer"]

# Define counter and busy/idle regions
counter_1_region = np.array([(739, 87), (848, 87), (848, 292), (739, 292)], dtype=np.int32)
counter_2_region = np.array([(969, 78), (1071, 78), (1071, 293), (969, 293)], dtype=np.int32)

busy_region_counter_1 = np.array([(608, 34), (929, 34), (929, 545), (608, 545)], dtype=np.int32)
busy_region_counter_2 = np.array([(945, 36), (1260, 36), (1260, 566), (945, 566)], dtype=np.int32)

counter_regions = {0: counter_1_region, 1: counter_2_region}
busy_regions = {0: busy_region_counter_1, 1: busy_region_counter_2}

# Variables for tracking
elapsed_time = defaultdict(float)  # Time spent by each customer
customer_count_per_counter = defaultdict(list)  # Customers who interacted with each counter
total_customers = set()  # Unique customers tracked across counters
counter_status = {cid: "Idle" for cid in counter_regions.keys()}  # Idle/Busy status
person_centroids = OrderedDict()  # Tracks customer centroids
original_fps = cap.get(cv2.CAP_PROP_FPS)

# Time spent in busy regions for each counter
time_in_busy_regions = defaultdict(float)
customers_in_busy_regions = defaultdict(set)

# Threshold for high or low activity
busy_threshold = 50  # seconds
activity_level = {0: "Low", 1: "Low"}  # Default activity level for counters

# Video writer setup
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("output/MAIN2.mp4", fourcc, original_fps, (1920, 1080))

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (1920, 1080))
    results = model.predict(frame, half=True, iou=0.5, conf=0.7)
    a = results[0].boxes.data.cpu()
    px = pd.DataFrame(a).astype("float")

    # Directly count active customers based on YOLO detections
    active_customers_count = (px[5] == class_list.index("Customer")).sum()

    customers_in_region = defaultdict(list)  # Customers in each counter's busy region

    for index, row in px.iterrows():
        x1, y1, x2, y2 = map(int, row[:4])
        d = int(row[5])
        c = class_list[d]

        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        if c == "Customer":
            # Track or register the customer
            min_distance = float('inf')
            closest_person_id = None
            for person_id, centroid in person_centroids.items():
                distance = np.sqrt((centroid[0] - center_x) ** 2 + (centroid[1] - center_y) ** 2)
                if distance < min_distance:
                    min_distance = distance
                    closest_person_id = person_id

            if min_distance < 100:
                # Update existing customer
                person_centroids[closest_person_id] = (center_x, center_y)
                elapsed_time[closest_person_id] += 1 / original_fps
            else:
                # Register a new customer
                person_id = len(person_centroids)
                person_centroids[person_id] = (center_x, center_y)
                elapsed_time[person_id] = 0
                total_customers.add(person_id)  # Track unique customers

            # Draw customer bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"Customer | Time: {elapsed_time[closest_person_id]:.2f}s",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

            # Check if customer is in any counter's busy region
            for cid, region in busy_regions.items():
                if cv2.pointPolygonTest(region, (center_x, center_y), False) >= 0:
                    customers_in_region[cid].append(closest_person_id)
                    time_in_busy_regions[cid] += 1 / original_fps
                    customers_in_busy_regions[cid].add(closest_person_id)

        elif c == "Counter":
            # Use pointPolygonTest to label the counter based on the region it lies in
            if cv2.pointPolygonTest(counter_1_region, (center_x, center_y), False) >= 0:
                label = "Counter 1"
            else:
                label = "Counter 2"

            # Draw counter bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2,
            )

    # Update counter statuses
    for cid, region in busy_regions.items():
        if customers_in_region[cid]:
            counter_status[cid] = "Busy"
            for person_id in customers_in_region[cid]:
                if person_id not in customer_count_per_counter[cid]:
                    customer_count_per_counter[cid].append(person_id)
        else:
            counter_status[cid] = "Idle"

        activity_level[cid] = "High" if time_in_busy_regions[cid] > busy_threshold else "Low"

    # Calculate average times for each counter
    avg_times = {}
    for cid in counter_regions.keys():
        total_time = sum(elapsed_time[person_id] for person_id in customers_in_busy_regions[cid])
        customer_count = len(customers_in_busy_regions[cid])
        avg_times[cid] = total_time / customer_count if customer_count > 0 else 0

    # Display active customers and counter information
    cv2.putText(
        frame,
        f"Active Customers: {active_customers_count}",
        (50, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )
    cv2.putText(
        frame,
        f"Total Customers: {len(total_customers)}",
        (50, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )

    y_offset = 110
    for cid, region in counter_regions.items():
        status_label = f"Counter {cid + 1} | {counter_status[cid]} | Avg Time: {avg_times[cid]:.2f}s"
        cv2.putText(frame, status_label, (50, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += 30

        # Display activity level separately
        activity_label = f"Activity: {activity_level[cid]}"
        cv2.putText(frame, activity_label, (50, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        y_offset += 30

    out.write(frame)
    cv2.imshow('Processed Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print("Processing complete. Output saved to output/MAIN2.mp4")
