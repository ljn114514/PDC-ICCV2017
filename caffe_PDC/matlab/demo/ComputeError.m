function alignment_errors = ComputeError( ground_truth_all, detected_points_all, face_rects )
%ComputeError
%   compute the average point-to-point Euclidean error normalized by the
%   inter-ocular distance (measured as the Euclidean distance between the
%   outer corners of the eyes)
%
%   Inputs:
%          grounth_truth_all, size: num_of_points x 2 x num_of_images
%          detected_points_all, size: num_of_points x 2 x num_of_images
%          face_rects, size: 4 x num_of_images
%   Output:
%          alignment_errors, size: num_of_images x 1

%------------------------------------------------------------------------
%   modification:
%   add pts_num = 31
%   @author: Xiaohu Shao
%   @data  : 19:22, 20140721
%   add pts_num = 77
%   @author: Xiaohu Shao
%   @data  : 14:28, 20140716
%------------------------------------------------------------------------

% use face width as normalization value or not
if nargin < 3
    use_face_rect = false;
else
    use_face_rect = true;
end

num_of_images = size(ground_truth_all,3);
num_of_points = size(ground_truth_all,1);

alignment_errors = zeros(num_of_images,1);

for i =1:num_of_images
    detected_points      = detected_points_all(:,:,i);
    ground_truth_points  = ground_truth_all(:,:,i);
    
    if ~use_face_rect
        if num_of_points == 77
            interocular_distance = norm(ground_truth_points(39,:)-ground_truth_points(40,:));
        elseif num_of_points == 68
            % modified by shao
            left_eye_center = mean(ground_truth_points(37:42, :));
            right_eye_center = mean(ground_truth_points(43:48, :));
            interocular_distance = norm(left_eye_center-right_eye_center);
            
            % interocular_distance = norm(ground_truth_points(37,:)-ground_truth_points(46,:));
        elseif num_of_points == 51
            interocular_distance = norm(ground_truth_points(20,:)-ground_truth_points(29,:));
        elseif num_of_points == 31
            interocular_distance = norm(ground_truth_points(06,:)-ground_truth_points(15,:));
        elseif num_of_points == 29
            interocular_distance = norm(ground_truth_points(05,:)-ground_truth_points(18,:));
        elseif num_of_points == 19
            interocular_distance = norm(ground_truth_points(07,1:2)-ground_truth_points(12,1:2));
        elseif num_of_points == 17
            interocular_distance = norm(ground_truth_points(03,:)-ground_truth_points(10,:));
        elseif num_of_points == 9
            interocular_distance = norm(ground_truth_points(02,:)-ground_truth_points(05,:));
        elseif num_of_points == 7
            interocular_distance = norm(ground_truth_points(02,:)-ground_truth_points(05,:));
        end
    else
        interocular_distance = face_rects(3, i) - face_rects(1, i);
    end
    
    
    if num_of_points == 19
%         v_value = repmat(ground_truth_points(:, 3), [1, 2]); 
%         diff = (ground_truth_points(:, 1:2) - detected_points) .* v_value;
        diff = (ground_truth_points(:, 1:2) - detected_points) ;
    else 
        diff = ground_truth_points - detected_points;
    end

    dsum = sum(sqrt(sum(diff .^ 2, 2)));
    alignment_errors(i) = dsum/(num_of_points*interocular_distance);
end


end

