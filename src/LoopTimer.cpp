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
