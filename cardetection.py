import cv2 as cv

count = 0
plate_cascade = cv.CascadeClassifier('haarcascade_russian_plate_number.xml')

cap = cv.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

while True:
    success, frame = cap.read()
    if not success:
        print("Failed to read frame")
        break
    
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    plates = plate_cascade.detectMultiScale(frame_gray, scaleFactor=1.5, minNeighbors=5)

    for (x, y, w, h) in plates:
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv.putText(frame, 'License Plate', (x, y - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        frame_roi = frame[y:y + h, x:x + w]
        cv.imshow('License Plate ROI', frame_roi)

        if cv.waitKey(1) & 0xFF == ord('s'):
            cv.imwrite('D:/opencv/carplate detection/scanned/NoPlate_' + str(count) + ".jpg", frame_roi)
            cv.rectangle(frame, (0, 200), (640, 300), (0, 255, 0), cv.FILLED)
            cv.putText(frame, 'scan saved ', (150, 265), cv.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 2)
            cv.imshow('License Plate Detection', frame)
            cv.waitKey(500)
            count += 1

    cv.imshow('License Plate Detection', frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
