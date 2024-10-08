import cv2
from ultralytics import YOLO


model = YOLO("yolov8n.pt")
names = model.model.names
cap = cv2.VideoCapture("Road_1.mp4")
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
video_writer = cv2.VideoWriter("vehicle_counting_output.avi", cv2.VideoWriter_fourcc(*"XVID"), fps, (w, h))
line1_start = (0, int(h * 0.5))  
line1_end = (w, int(h * 0.5))
line2_start = (0, int(h * 0.7))  
line2_end = (w, int(h * 0.7))
car_count, truck_count, bus_count = 0, 0, 0
vehicle_directions = {}
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
    results = model.track(frame, persist=True, show=False)
    boxes = results[0].boxes if results else None
    if boxes:
        for box in boxes:
            cls_id = int(box.cls)  
            label = names[cls_id] 
            bbox_center_y = int((box.xyxy[0][1] + box.xyxy[0][3]) / 2)
            if bbox_center_y <= line1_start[1]:  
                if box.id not in vehicle_directions:
                    vehicle_directions[box.id] = 'unknown'  
            elif line1_start[1] < bbox_center_y <= line2_start[1]:  
                if box.id not in vehicle_directions:
                    vehicle_directions[box.id] = 'crossed line1'  
            elif bbox_center_y > line2_start[1]:  
                if box.id in vehicle_directions and vehicle_directions[box.id] == 'crossed line1':
                    if label == 'car':
                        car_count += 1
                    elif label == 'truck':
                        truck_count += 1
                    elif label == 'bus':
                        bus_count += 1
                    vehicle_directions[box.id] = 'counted down' 
                elif box.id not in vehicle_directions:
                    vehicle_directions[box.id] = 'crossed line2' 
    cv2.line(frame, line1_start, line1_end, (0, 0, 255), 2)  # Red line 
    cv2.line(frame, line2_start, line2_end, (255, 0, 0), 2)  # Blue line 
    cv2.putText(frame, f'Cars: {car_count}', (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f'Trucks: {truck_count}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f'Buses: {bus_count}', (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    video_writer.write(frame)
cap.release()
video_writer.release()

