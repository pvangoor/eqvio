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

#include "eqvio/LoopTimer.h"

void LoopTimer::startTiming(const std::string& label) { timerStartPoints.at(label) = timer_clock::now(); }

void LoopTimer::endTiming(const std::string& label) {
    timer_clock::time_point now = timer_clock::now();
    currentLoopTimingData.timings[label] = now - timerStartPoints.at(label);
}

void LoopTimer::startLoop() {

    // Reset all timing
    for (const auto& [label, timing] : timerStartPoints) {
        currentLoopTimingData.timings[label] = timer_duration(0);
    }

    // Reset the loop timing data
    currentLoopTimingData.loopTimeStart = timer_clock::now() - timerOrigin;
}

void LoopTimer::initialise(const std::vector<std::string>& headers) {
    const timer_clock::time_point now = timer_clock::now();
    for (const std::string& header : headers) {
        timerStartPoints[header] = now;
    }
}
