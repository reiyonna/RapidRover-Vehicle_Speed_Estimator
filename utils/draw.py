import cv2

font = cv2.FONT_HERSHEY_COMPLEX

def draw_text_with_background(img, text, pos):
    font_scale = 1
    x,y = pos
    text_size, _ = cv2.getTextSize(text, font, font_scale, 2)
    text_w, text_h = text_size
    padding_x, padding_y = 10, 10
    img2 = cv2.rectangle(img, pos, (x + text_w + padding_x, y + text_h + padding_y), (48, 48, 48), -1)
    frame = cv2.putText(img2, text,  (x, y + text_h + font_scale - 1), font, font_scale, (255, 255, 255), 2, cv2.LINE_AA)

    return frame