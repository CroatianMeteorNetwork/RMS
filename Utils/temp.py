    def computeIntensitySum(self, photom_pixels, global_centroid, raw_frame):
        """ Compute the background subtracted sum of intensity of colored pixels. The background is estimated
            as the median of near pixels that are not colored.
        """
            # IMPLEMENT:
                # self.dark (IF AVAIL)
                # self.flat_struc (IF AVAIL)
                # self.skyfit_config

            x_arr, y_arr = np.array(photom_pixels).T

            # Take a window twice the size of the colored pixels
            x_color_size = np.max(x_arr) - np.min(x_arr)
            y_color_size = np.max(y_arr) - np.min(y_arr)

            x_min = int(global_centroid[0] - x_color_size)
            x_max = int(global_centroid[0] + x_color_size)
            y_min = int(global_centroid[1] - y_color_size)
            y_max = int(global_centroid[1] + y_color_size)

            # Limit the size to be within the bounds
            if x_min < 0: x_min = 0
            if x_max > raw_frame.shape[0]: x_max = raw_frame.shape[0]
            if y_min < 0: y_min = 0
            if y_max > raw_frame.shape[1]: y_max = raw_frame.shape[1]

            # Take only the colored part
            mask_img = np.ones_like(raw_frame)
            mask_img[x_arr, y_arr] = 0
            masked_img = np.ma.masked_array(raw_frame, mask_img)
            crop_img = masked_img[x_min:x_max, y_min:y_max]

            # Perform gamma correction on the colored part
            crop_img = Image.gammaCorrectionImage(crop_img, self.skyfit_config.gamma, bp=0, wp=(2**self.skyfit_config.bit_depth - 1))

            # Mask out the colored in pixels
            mask_img_bg = np.zeros_like(raw_frame)
            mask_img_bg[x_arr, y_arr] = 1

            # Take the image where the colored part is masked out and crop the surroundings
            masked_img_bg = np.ma.masked_array(raw_frame, mask_img_bg)
            crop_bg = masked_img_bg[x_min:x_max, y_min:y_max]

            # Perform gamma correction on the background
            crop_bg = Image.gammaCorrectionImage(crop_bg, self.skyfit_config.gamma, bp=0, wp=(2**self.skyfit_config.bit_depth - 1))

            # Compute the median background
            background_lvl = np.ma.median(crop_bg)

                    # # If the DFN image is used and a dark has been applied (i.e. the previous image is subtracted),
                    # #   assume that the background is zero
                    # if (self.img_handle.input_type == "dfn") and (self.dark is not None):
                    #     background_lvl = 0

                    # # If the nobg flag is set, assume that the background is zero.
                    # # This is useful when the background is already subtracted or saturated objects are being
                    # #  measured
                    # if self.no_background_subtraction:
                    #     background_lvl = 0


                    # # If the background level is set to zero, simply sum up the intensity of the colored pixels
                    # if background_lvl == 0:
                    #     intensity_sum = np.ma.sum(crop_img)

                    # # Use the peripheral background subtraction method if forced or on static images
                    # elif self.peripheral_background_subtraction \
                    #     or (self.img_handle.input_type == "dfn") \
                    #     or (hasattr(self.img_handle, "single_image_mode") and  self.img_handle.single_image_mode):

                    #     # Compute the background subtracted intensity sum by using pixels peripheral to the colored 
                    #     # pixels
                    #     # (do as a float to avoid artificially pumping up the magnitude)
                    #     crop_img_nobg = crop_img.astype(float) - background_lvl
                    #     crop_img_nobg = np.clip(crop_img_nobg, 0, None)
                    #     intensity_sum = np.ma.sum(crop_img_nobg)

            # Subtract the background using the avepixel
            else:

                # Get the avepixel image and apply a dark and flat to it if needed
                avepixel = self.img_handle.ff.avepixel.T

                if self.dark is not None:
                    avepixel = applyDark(avepixel, self.dark)
                if self.flat_struct is not None:
                    avepixel = applyFlat(avepixel, self.flat_struct)

                
                # Mask out the colored in pixels
                avepixel_masked = np.ma.masked_array(avepixel, mask_img)
                avepixel_crop = avepixel_masked[x_min:x_max, y_min:y_max]

                # Perform gamma correction on the avepixel crop
                avepixel_crop = Image.gammaCorrectionImage(avepixel_crop, self.config.gamma, bp=0, wp=(2**self.config.bit_depth - 1))

                background_lvl = np.ma.median(avepixel_crop)

                # Subtract the avepixel crop from the data crop, clip the negative values to 0 and
                #  sum up the intensity
                crop_img_nobg = crop_img.astype(float) - avepixel_crop
                crop_img_nobg = np.clip(crop_img_nobg, 0, None)
                intensity_sum = np.ma.sum(crop_img_nobg)

            # Check if the result is masked
            if np.ma.is_masked(intensity_sum):
                # If the result is masked (i.e. error reading pixels), set the intensity sum to 1
                print("Warning: intensity sum is masked, setting to 1") #Fallback to regular method, ENSURE NEVER 0
                intensity_sum = 1
            else:
                intensity_sum = intensity_sum.astype(int)

            ### Measure the SNR of the pick ###

            # Compute the standard deviation of the background
            background_stddev = np.ma.std(crop_bg)

            # Count the number of pixels in the photometric area
            source_px_count = np.ma.sum(~crop_img.mask)

            # Compute the signal to noise ratio using the CCD equation
            snr = signalToNoise(intensity_sum, source_px_count, background_lvl, background_stddev)

            ### Determine if there is any saturation in the measured photometric area

            # Compute the saturation threshold
            saturation_threshold = int(0.98*(2**self.config.bit_depth))

            # If at least 2 pixels are saturated in the photometric area, mark the pick as saturated
            if np.sum(crop_img > saturation_threshold) >= 2:
                saturated_bool = True
            else:
                saturated_bool = False

            #


                    # # If the DFN image is used, correct intensity sum for exposure difference
                    # # Of the total 27 second, the stars are exposed 4.31 seconds, and every fireball dot is exposed
                    # #    a total of 0.01 seconds. Thus the correction factor is 431
                    # if (self.img_handle.input_type == "dfn"):
                    #     pick['intensity_sum'] *= 431

