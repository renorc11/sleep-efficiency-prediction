"use client";

import { useState } from "react";

export type LifestylePayload = {
  caffeine_mg_evening: number;
  alcohol_units: number;
  dinner_hour: number;
  stress_level: number;
  mood_score: number;
  physical_activity_min: number;
  naps_count: number;
  naps_total_min: number;
  previous_sleep_hours: number;
  work_hours: number;
  blue_light_minutes: number;
  bedtime_consistency: number;
  hydration_liters: number;
};

const defaults: LifestylePayload = {
  caffeine_mg_evening: 120,
  alcohol_units: 1,
  dinner_hour: 20,
  stress_level: 5,
  mood_score: 0,
  physical_activity_min: 40,
  naps_count: 1,
  naps_total_min: 20,
  previous_sleep_hours: 7,
  work_hours: 8,
  blue_light_minutes: 180,
  bedtime_consistency: 0.7,
  hydration_liters: 2.2
};

function NumberField({
  label,
  value,
  step = 1,
  min,
  max,
  disabled,
  onChange
}: {
  label: string;
  value: number;
  step?: number;
  min?: number;
  max?: number;
  disabled?: boolean;
  onChange: (v: number) => void;
}) {
  return (
    <div>
      <label>{label}</label>
      <input
        type="number"
        value={value}
        step={step}
        min={min}
        max={max}
        disabled={disabled}
        onFocus={(e) => e.currentTarget.select()}
        onChange={(e) => onChange(Number(e.target.value))}
      />
    </div>
  );
}

export default function LifestyleForm({ onSubmit, loading }: { onSubmit: (p: LifestylePayload) => void; loading: boolean }) {
  const [values, setValues] = useState<LifestylePayload>(defaults);
  const [hadCaffeine, setHadCaffeine] = useState(values.caffeine_mg_evening > 0);
  const [hadAlcohol, setHadAlcohol] = useState(values.alcohol_units > 0);
  const [activityUnit, setActivityUnit] = useState<"minutes" | "hours">("minutes");
  const [blueLightUnit, setBlueLightUnit] = useState<"minutes" | "hours">("minutes");

  const activityInputValue =
    activityUnit === "hours" ? values.physical_activity_min / 60 : values.physical_activity_min;
  const blueLightInputValue =
    blueLightUnit === "hours" ? values.blue_light_minutes / 60 : values.blue_light_minutes;

  function submit() {
    onSubmit({
      ...values,
      caffeine_mg_evening: hadCaffeine ? values.caffeine_mg_evening : 0,
      alcohol_units: hadAlcohol ? values.alcohol_units : 0
    });
  }

  return (
    <div className="card">
      <h2>Lifestyle Inputs</h2>
      <div className="grid">
        <div>
          <label>Caffeine consumed?</label>
          <select
            value={hadCaffeine ? "yes" : "no"}
            onChange={(e) => setHadCaffeine(e.target.value === "yes")}
          >
            <option value="yes">Yes</option>
            <option value="no">No</option>
          </select>
        </div>
        <NumberField
          label="Caffeine quantity (mg, evening)"
          value={values.caffeine_mg_evening}
          min={0}
          max={800}
          disabled={!hadCaffeine}
          onChange={(v) => setValues((prev) => ({ ...prev, caffeine_mg_evening: v }))}
        />

        <div>
          <label>Alcohol consumed?</label>
          <select
            value={hadAlcohol ? "yes" : "no"}
            onChange={(e) => setHadAlcohol(e.target.value === "yes")}
          >
            <option value="yes">Yes</option>
            <option value="no">No</option>
          </select>
        </div>
        <NumberField
          label="Alcohol quantity (units)"
          value={values.alcohol_units}
          step={0.5}
          min={0}
          max={10}
          disabled={!hadAlcohol}
          onChange={(v) => setValues((prev) => ({ ...prev, alcohol_units: v }))}
        />

        <NumberField
          label="Dinner hour (24h format)"
          value={values.dinner_hour}
          step={0.5}
          min={16}
          max={24}
          onChange={(v) => setValues((prev) => ({ ...prev, dinner_hour: v }))}
        />
        <NumberField
          label="Stress level (1-10)"
          value={values.stress_level}
          min={1}
          max={10}
          onChange={(v) => setValues((prev) => ({ ...prev, stress_level: v }))}
        />
        <NumberField
          label="Mood score (-2 to 2)"
          value={values.mood_score}
          step={0.1}
          min={-2}
          max={2}
          onChange={(v) => setValues((prev) => ({ ...prev, mood_score: v }))}
        />

        <div>
          <label>Physical activity unit</label>
          <select
            value={activityUnit}
            onChange={(e) => setActivityUnit(e.target.value as "minutes" | "hours")}
          >
            <option value="minutes">Minutes</option>
            <option value="hours">Hours</option>
          </select>
        </div>
        <NumberField
          label={`Physical activity (${activityUnit})`}
          value={activityInputValue}
          step={activityUnit === "hours" ? 0.1 : 1}
          min={0}
          onChange={(v) =>
            setValues((prev) => ({
              ...prev,
              physical_activity_min: activityUnit === "hours" ? v * 60 : v
            }))
          }
        />

        <NumberField
          label="Naps count"
          value={values.naps_count}
          min={0}
          max={5}
          onChange={(v) => setValues((prev) => ({ ...prev, naps_count: Math.max(0, Math.round(v)) }))}
        />
        <NumberField
          label="Naps total (minutes)"
          value={values.naps_total_min}
          min={0}
          max={240}
          onChange={(v) => setValues((prev) => ({ ...prev, naps_total_min: v }))}
        />
        <NumberField
          label="Previous sleep (hours)"
          value={values.previous_sleep_hours}
          step={0.1}
          min={3}
          max={12}
          onChange={(v) => setValues((prev) => ({ ...prev, previous_sleep_hours: v }))}
        />
        <NumberField
          label="Work hours"
          value={values.work_hours}
          step={0.1}
          min={0}
          max={16}
          onChange={(v) => setValues((prev) => ({ ...prev, work_hours: v }))}
        />

        <div>
          <label>Blue-light exposure unit</label>
          <select
            value={blueLightUnit}
            onChange={(e) => setBlueLightUnit(e.target.value as "minutes" | "hours")}
          >
            <option value="minutes">Minutes</option>
            <option value="hours">Hours</option>
          </select>
        </div>
        <NumberField
          label={`Blue-light exposure (${blueLightUnit})`}
          value={blueLightInputValue}
          step={blueLightUnit === "hours" ? 0.1 : 1}
          min={0}
          onChange={(v) =>
            setValues((prev) => ({
              ...prev,
              blue_light_minutes: blueLightUnit === "hours" ? v * 60 : v
            }))
          }
        />

        <NumberField
          label="Bedtime consistency (0-1)"
          value={values.bedtime_consistency}
          step={0.05}
          min={0}
          max={1}
          onChange={(v) => setValues((prev) => ({ ...prev, bedtime_consistency: v }))}
        />
        <NumberField
          label="Hydration (liters)"
          value={values.hydration_liters}
          step={0.1}
          min={0.5}
          max={6}
          onChange={(v) => setValues((prev) => ({ ...prev, hydration_liters: v }))}
        />
        <div>
          <label>&nbsp;</label>
          <p className="small">Click any number field and type directly; current value is auto-selected.</p>
        </div>
        <div>
          <label>&nbsp;</label>
          <p className="small">Backend still receives activity and blue-light values in minutes.</p>
        </div>
      </div>
      <div style={{ marginTop: 12 }}>
        <button onClick={submit} disabled={loading}>{loading ? "Predicting..." : "Predict Sleep Efficiency"}</button>
      </div>
    </div>
  );
}
