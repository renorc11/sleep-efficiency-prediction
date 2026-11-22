"use client";

import { useEffect, useMemo, useState } from "react";

const BINS = 288;
const MS_PER_BIN = 5 * 60 * 1000;

function nowToBin() {
  const d = new Date();
  return Math.floor((d.getHours() * 60 + d.getMinutes()) / 5);
}

export function useScreenActivitySequence() {
  const [seq, setSeq] = useState<number[]>(() => {
    const init = new Array(BINS).fill(0);
    init[nowToBin()] = 1;
    return init;
  });

  useEffect(() => {
    const markActive = () => {
      setSeq((prev) => {
        const next = [...prev];
        next[nowToBin()] = 1;
        return next;
      });
    };

    const events: Array<keyof WindowEventMap> = ["mousemove", "keydown", "touchstart", "click"];
    events.forEach((e) => window.addEventListener(e, markActive));
    const timer = setInterval(markActive, 60_000);

    return () => {
      events.forEach((e) => window.removeEventListener(e, markActive));
      clearInterval(timer);
    };
  }, []);

  const inactivitySummary = useMemo(() => {
    let best = 0;
    let current = 0;
    for (const v of seq) {
      if (v === 0) {
        current += 1;
        best = Math.max(best, current);
      } else {
        current = 0;
      }
    }
    return `${best * 5} min longest inactivity window today`;
  }, [seq]);

  return { sequence: seq, inactivitySummary };
}

export default function ScreenTracker({ summary }: { summary: string }) {
  return (
    <div className="small">
      <p style={{ margin: "0 0 4px" }}>Screen behavior tracking active: {summary}</p>
      <p className="disclosure" style={{ marginTop: 0 }}>
        Activity is recorded only from this browser tab, in five-minute bins from midnight, and is sent with your
        prediction. Inactivity is an indirect behavioral proxy, not a measurement of sleep. The models were trained on
        synthetic activity sequences and have not been validated on real browser or phone activity.
      </p>
    </div>
  );
}
