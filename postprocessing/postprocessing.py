def postprocessing(source_image, prediction):
    """
    """
    source_image_resized = skimage.transform.rescale(source_image, (1/4,1/4), True)
    source_image_resized_bw = rgb2gray(source_image_resized)
    
    source_edges = feature.canny(source_image_resized_bw, 1.5)
    
    prediction_resized = skimage.transform.rescale(prediction, (1/4,1/4), True)
    prediction_edges = feature.canny(prediction_resized, 1.5)
    
    hough_lines_source = probabilistic_hough_line(edges, line_length=8,line_gap=2)

    cut = np.zeros(edges.shape)
    img = np.zeros(edges.shape)

    for line in hough_lines_source:
        p0, p1 = line
        main_line = skimage.draw.line(line[1][1], line[1][0], line[0][1], line[0][0])

        # draw 3 parallel lines in both direction and look at how many edge pixels are around
        for i in range(3):
            if(p0[1]+i <edges.shape[0] and p1[1]+i<edges.shape[1] and p0[0]+i <edges.shape[0] and p1[0]+i<edges.shape[1]):
                
                line = skimage.draw.line(p1[1]+i, p1[0], p0[1]+i, p0[0])
                if(np.count_nonzero(prediction_edges[line])>7):
                    
                    # if the line is near to edge pixels, we assume that the line is a border and 
                    # align the edge pixel with it
                    p = np.sum(prediction_edges[line])/len(prediction_edges[line])/2
                    prediction_resized[line] = [min(1, v+p) for v in prediction_resized[line]]
                
                line = skimage.draw.line(p1[1]+i, p1[0], p0[1]+i, p0[0])
                if(np.count_nonzero(prediction_edges[line])>7):
                    
                    # if the line is near to edge pixels, we assume that the line is a border and 
                    # align the edge pixel with it
                    p = np.sum(prediction_edges[line])/np.count_nonzero([prediction_edges[line]])
                    for pixel in prediction_resized[line]:
                        p = np.sum(prediction_edges[line])/len(prediction_edges[line])
                        prediction_resized[line] = [min(1, v+p) for v in prediction_resized[line]]
    
    return skimage.transform.rescale(prediction_resized, (4,4), True)