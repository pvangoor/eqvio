/*
    This file is part of EqVIO.

    EqVIO is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    EqVIO is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with EqVIO.  If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once

#include "eqvio/aofstream.h"

#include <chrono>
#include <map>
#include <string>
#include <vector>

/** @brief A class for timing the processing speed of code sections in a loop.
 *
 * This class is intended to be instantiated as a global loop timer variable. It is set up with some initial timing
 * category labels. Each loop should be denoted with a call to startLoop. Then, the user can call startTiming and
 * endTiming with any of the category labels to start and stop a timer for a given section of code. At the end of each
 * loop, the total time spent on each category of code can be obtained from getLoopTimingData.
 */
class LoopTimer {
  public:
    /// Shorthand for the clock used
    using timer_clock = std::chrono::steady_clock;
    /// Shorthand for the duration type
    using timer_duration = std::chrono::duration<double>;

    /// @brief a struct to contain a time stamp and map from labels to timer duration
    struct LoopTimingData {
        timer_duration loopTimeStart;                  ///< The time when the loop was started
        std::map<std::string, timer_duration> timings; ///< The time taken for each section of code in the loop
    };

  protected:
    /// The time at which the loop timer is created.
    const timer_clock::time_point timerOrigin = timer_clock::now();

    /// The initial time for each label.
    std::map<std::string, timer_clock::time_point> timerStartPoints;
    /// The latest loop timing data.
    LoopTimingData currentLoopTimingData;

  public:
    /** @brief start a timing loop.
     *
     * Reset all the current elapsed times to zero, and all the current timer start points to now. This should be called
     * at the start of every loop that is to be timed.
     */
    void startLoop();

    /** @brief start timing the code associated with the given label.
     *
     * @param label The label of the relevant category.
     *
     * Records the current time in the timerStartPoints.
     */

    void startTiming(const std::string& label);
    /** @brief stop timing the code associated with the given label.
     *
     * @param label The label of the relevant category.
     *
     * Subtracts the timer start point of this label from the current time.
     */
    void endTiming(const std::string& label);

    /** @brief set the labels that will be used later for recording data.
     *
     * @param headers A std::vector of labels for timing data.
     *
     * Once initialised, no additional labels should be added.
     */
    void initialise(const std::vector<std::string>& headers);

    /** @brief return the current loop timing data.
     *
     * Should be used at the end of each loop that is to be timed.
     */
    const LoopTimingData& getLoopTimingData() const { return currentLoopTimingData; }
};

inline LoopTimer loopTimer;