    def pred_to_rgb(self, pred, color_table, PT):
        unique_class = np.unique(pred)
        bounding_box = []
        
        for class_id in unique_class:
            class_id = int(class_id)
            if class_id ==1 or class_id ==2 or class_id ==3 or class_id ==4 or class_id ==5 or class_id ==6 or class_id ==7:
                class_prob_map = pred[class_id]            
                class_mask = (class_prob_map >= PT[CLASSES[class_id]]).astype(np.uint8)
                contours, _ = cv2.findContours(class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    x, y, w, h = cv2.boundingRect(contour)
                    if w * h <40:
                        continue
                    bounding_box.append((class_id, (int(x*self.scale_x), int(y*self.scale_y), int(w*self.scale_x), int(h*self.scale_y))))                
            else:
                pass
      
        arg_pred = np.argmax(pred, axis=0)

        pred_rgb = np.zeros_like(arg_pred, dtype=np.uint8)
        pred_rgb = np.repeat(np.expand_dims(pred_rgb[:, :], axis=-1), 3, -1)
        s_time = time.time()       
        for i in range(len(CLASSES)):
            if CLASSES[i] == 'constructionGuide' or CLASSES[i] == 'warningTriangle':
                pass
            else:
                pred_rgb[arg_pred ==i] = np.array(color_table[i])  
        e_time = time.time()
        #print(1 / (e_time - s_time))              
        return pred_rgb, bounding_box
