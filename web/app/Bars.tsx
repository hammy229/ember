// A horizontal bar chart for saved measurements. Bars are scaled to the largest value, and
// baseline rows (what Ember is compared against) are drawn in slate.

export type BarRow = {
  name: string;
  value: number;
  label: string;
  baseline?: boolean;
};

export function Bars({ caption, rows }: { caption: string; rows: BarRow[] }) {
  const largest = Math.max(...rows.map((row) => row.value));
  return (
    <figure className="bars">
      <figcaption>{caption}</figcaption>
      <dl>
        {rows.map((row) => (
          <div key={row.name}>
            <dt>{row.name}</dt>
            <dd>
              <span
                className={row.baseline ? "bar is-baseline" : "bar"}
                style={{
                  width: `calc((100% - 76px) * ${row.value / largest})`,
                }}
              />
              <span className="bar-value">{row.label}</span>
            </dd>
          </div>
        ))}
      </dl>
    </figure>
  );
}
