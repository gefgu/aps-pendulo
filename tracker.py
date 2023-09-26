import cv2
import pandas as pd

video = cv2.VideoCapture("./pendulo.mp4")

object_detector = cv2.createBackgroundSubtractorKNN(history=3000, dist2Threshold=900, detectShadows=False)

data = []
fps = video.get(cv2.CAP_PROP_FPS)

t = 0

while True:
  working, frame = video.read()
  if(not working):
    break

  frame = cv2.rotate(frame, cv2.ROTATE_180)
  frame = cv2.resize(frame, (540, 960))

  mask = object_detector.apply(frame)

  contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > 200:
      #cv2.drawContours(frame, [cnt], -1, (0, 255, 0), 2)
      x, y, w, h = cv2.boundingRect(cnt)
      cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
      cv2.putText(frame, f"({int(x + w/2)}, {int(y + h/2)})", (x - 15, y-15), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)

      data.append({"t": t, "x": (x + w/2)})

      break
      

  cv2.imshow("Frame", frame)
  cv2.imshow("Mask", mask)
  t += 1.0 / fps
  print(f"Time: {t}")

  key = cv2.waitKey(30)
  if key == 27: # ESC
    break

video.release()
cv2.destroyAllWindows()

df = pd.DataFrame(data)
df.to_csv("positions.csv", index=False)