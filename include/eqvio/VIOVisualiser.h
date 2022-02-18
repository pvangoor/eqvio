#pragma once

#include "GIFT/Feature.h"
#include "Plotter.h"
#include "eqvio/VIOState.h"

/** @brief A class to handle visualisation of VIO state
 *
 * This includes drawing feature points onto the current video frame and also showing the current VIO state in a 3D
 * rendering. The 3D rendering is only available if the program is built with support enabled.
 */
class VIOVisualiser {
  protected:
#if EQVIO_BUILD_VISUALISATION
    std::map<int, int> pointLifetimeCounter;         ///< The number of frames each point has been seen by id number.
    std::map<int, Eigen::Vector3d> persistentPoints; ///< The points that have existed for a sufficiently long time.
    std::vector<Eigen::Vector3d> positionTrail;      ///< The history of all robot positions over time.

    std::unique_ptr<Plotter> plotter; ///< A plotter instance to use when drawing the VIO state.
#endif

  public:
    /** @brief Update the 3D render of the VIO system from the current state estimate.
     *
     * @param state The state to be added to the map display.
     */
    void updateMapDisplay(const VIOState& state);

    /** @brief Displays the features as points on an image.
     *
     * @param features The image feature points to be displayed.
     * @param baseImage The image on which features should be drawn.
     */
    void displayFeatureImage(const std::vector<GIFT::Feature>& features, const cv::Mat& baseImage);

    /** @brief Construct the VIO visualiser with optional display
     *
     * @param displayFlag Create displays if true.
     *
     * @todo The display flag is not sensible. It was introduced to make display optional, but really, if the user does
     * not want a display then they should not instantiate a visualiser in the first place.
     */
    VIOVisualiser(const bool& displayFlag = false) {
        if (displayFlag) {
            plotter = std::make_unique<Plotter>();
        }
    }

    /** @brief Destroy the visualiser, but first join the plotter thread if one exists.
     */
    ~VIOVisualiser();
};