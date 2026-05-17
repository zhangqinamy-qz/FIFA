import { useEffect, useMemo, useState } from 'react'

const COUNTRY_FLAG = {
  Mexico: '🇲🇽', 'South Africa': '🇿🇦', 'South Korea': '🇰🇷', 'Czech Republic': '🇨🇿',
  Canada: '🇨🇦', 'Bosnia and Herzegovina': '🇧🇦', Qatar: '🇶🇦', Switzerland: '🇨🇭',
  Brazil: '🇧🇷', Morocco: '🇲🇦', Haiti: '🇭🇹', Scotland: '🏴󠁧󠁢󠁳󠁣󠁴󠁿',
  'United States': '🇺🇸', Paraguay: '🇵🇾', Australia: '🇦🇺', Turkey: '🇹🇷',
  Germany: '🇩🇪', Curaçao: '🇨🇼', 'Ivory Coast': '🇨🇮', Ecuador: '🇪🇨',
  Netherlands: '🇳🇱', Japan: '🇯🇵', Sweden: '🇸🇪', Tunisia: '🇹🇳',
  Belgium: '🇧🇪', Egypt: '🇪🇬', Iran: '🇮🇷', 'New Zealand': '🇳🇿',
  Spain: '🇪🇸', 'Cape Verde': '🇨🇻', 'Saudi Arabia': '🇸🇦', Uruguay: '🇺🇾',
  France: '🇫🇷', Senegal: '🇸🇳', Iraq: '🇮🇶', Norway: '🇳🇴',
  Argentina: '🇦🇷', Algeria: '🇩🇿', Austria: '🇦🇹', Jordan: '🇯🇴',
  Portugal: '🇵🇹', 'DR Congo': '🇨🇩', Uzbekistan: '🇺🇿', Colombia: '🇨🇴',
  England: '🏴󠁧󠁢󠁥󠁮󠁧󠁿', Croatia: '🇭🇷', Ghana: '🇬🇭', Panama: '🇵🇦',
}

const flag = (t) => COUNTRY_FLAG[t] || '⚽'

const pct = (x) => `${(x * 100).toFixed(1)}%`
const barColor = (p) => {
  if (p >= 0.5) return 'bg-emerald-500'
  if (p >= 0.25) return 'bg-amber-500'
  if (p >= 0.10) return 'bg-orange-500'
  return 'bg-rose-700/70'
}

function ProbBar({ value, max = 1, color }) {
  const w = Math.min(100, (value / max) * 100)
  return (
    <div className="w-24 h-2 bg-neutral-800 rounded overflow-hidden">
      <div className={`h-full ${color || barColor(value)}`} style={{ width: `${w}%` }} />
    </div>
  )
}

function TopContenders({ teams }) {
  const top = [...teams].sort((a, b) => b.p_winner - a.p_winner).slice(0, 16)
  return (
    <section>
      <h2 className="text-lg font-semibold text-neutral-300 mb-3 tracking-wide uppercase">Title contenders</h2>
      <div className="overflow-x-auto rounded-lg border border-neutral-800">
        <table className="w-full text-sm">
          <thead className="bg-neutral-900 text-neutral-400 uppercase text-xs">
            <tr>
              <th className="text-left px-3 py-2 font-medium">#</th>
              <th className="text-left px-3 py-2 font-medium">Team</th>
              <th className="text-center px-3 py-2 font-medium">Group</th>
              <th className="text-right px-3 py-2 font-medium">R16</th>
              <th className="text-right px-3 py-2 font-medium">QF</th>
              <th className="text-right px-3 py-2 font-medium">SF</th>
              <th className="text-right px-3 py-2 font-medium">Final</th>
              <th className="text-right px-3 py-2 font-medium">Winner</th>
              <th className="text-left px-3 py-2 font-medium w-32">P(winner)</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-neutral-800">
            {top.map((t, i) => (
              <tr key={t.team} className="hover:bg-neutral-900/60">
                <td className="px-3 py-2 text-neutral-500">{i + 1}</td>
                <td className="px-3 py-2 font-medium">
                  <span className="mr-2">{flag(t.team)}</span>{t.team}
                </td>
                <td className="px-3 py-2 text-center text-neutral-400">{t.group}</td>
                <td className="px-3 py-2 text-right tabular-nums text-neutral-300">{pct(t.p_R16)}</td>
                <td className="px-3 py-2 text-right tabular-nums text-neutral-300">{pct(t.p_QF)}</td>
                <td className="px-3 py-2 text-right tabular-nums text-neutral-300">{pct(t.p_SF)}</td>
                <td className="px-3 py-2 text-right tabular-nums text-neutral-300">{pct(t.p_final)}</td>
                <td className="px-3 py-2 text-right tabular-nums font-semibold">{pct(t.p_winner)}</td>
                <td className="px-3 py-2">
                  <ProbBar value={t.p_winner} max={Math.max(...top.map((x) => x.p_winner))} color="bg-emerald-500" />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  )
}

function GroupCard({ group, teams, summary }) {
  const sorted = teams
    .map((t) => summary.find((s) => s.team === t))
    .filter(Boolean)
    .sort((a, b) => b.p_R16 - a.p_R16)
  return (
    <div className="rounded-lg border border-neutral-800 bg-neutral-900/50">
      <div className="px-3 py-2 border-b border-neutral-800 flex items-center justify-between">
        <span className="text-xs uppercase tracking-wider text-neutral-400">Group {group}</span>
        <span className="text-xs text-neutral-500">top-2 advance</span>
      </div>
      <ul className="divide-y divide-neutral-800">
        {sorted.map((t) => (
          <li key={t.team} className="px-3 py-2 flex items-center gap-3">
            <span className="text-lg">{flag(t.team)}</span>
            <span className="flex-1 truncate">{t.team}</span>
            <span className="tabular-nums text-xs text-neutral-400 w-12 text-right">{pct(t.p_R16)}</span>
            <ProbBar value={t.p_R16} />
          </li>
        ))}
      </ul>
    </div>
  )
}

function GroupGrid({ groups, summary }) {
  const letters = Object.keys(groups).sort()
  return (
    <section>
      <h2 className="text-lg font-semibold text-neutral-300 mb-3 tracking-wide uppercase">Groups</h2>
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-3">
        {letters.map((g) => (
          <GroupCard key={g} group={g} teams={groups[g]} summary={summary} />
        ))}
      </div>
    </section>
  )
}

function MatchupPicker({ teams, matchups }) {
  const teamNames = useMemo(() => [...teams.map((t) => t.team)].sort(), [teams])
  const [a, setA] = useState(teamNames[0])
  const [b, setB] = useState(teamNames[1])

  const found = useMemo(() => {
    if (a === b) return null
    return (
      matchups.find((m) => m.team_a === a && m.team_b === b) ||
      matchups.find((m) => m.team_a === b && m.team_b === a)
    )
  }, [a, b, matchups])

  const reversed = found && found.team_a === b
  const pAW = found ? (reversed ? found.p_b_wins : found.p_a_wins) : 0
  const pBW = found ? (reversed ? found.p_a_wins : found.p_b_wins) : 0
  const pD = found ? found.p_draw : 0

  return (
    <section>
      <h2 className="text-lg font-semibold text-neutral-300 mb-3 tracking-wide uppercase">Matchup probability (neutral venue)</h2>
      <div className="rounded-lg border border-neutral-800 bg-neutral-900/50 p-4">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3 mb-4">
          <select
            value={a}
            onChange={(e) => setA(e.target.value)}
            className="bg-neutral-950 border border-neutral-700 rounded px-3 py-2 text-sm"
          >
            {teamNames.map((t) => (
              <option key={t} value={t}>{flag(t)} {t}</option>
            ))}
          </select>
          <select
            value={b}
            onChange={(e) => setB(e.target.value)}
            className="bg-neutral-950 border border-neutral-700 rounded px-3 py-2 text-sm"
          >
            {teamNames.map((t) => (
              <option key={t} value={t}>{flag(t)} {t}</option>
            ))}
          </select>
        </div>
        {a === b ? (
          <p className="text-neutral-500 text-sm">Pick two different teams.</p>
        ) : !found ? (
          <p className="text-neutral-500 text-sm">No matchup data.</p>
        ) : (
          <div className="space-y-3">
            <div className="flex items-center gap-3">
              <span className="w-44 truncate font-medium">{flag(a)} {a} wins</span>
              <span className="tabular-nums w-16 text-right text-emerald-400">{pct(pAW)}</span>
              <div className="flex-1 h-3 bg-neutral-800 rounded overflow-hidden">
                <div className="h-full bg-emerald-500" style={{ width: `${pAW * 100}%` }} />
              </div>
            </div>
            <div className="flex items-center gap-3">
              <span className="w-44 truncate font-medium text-neutral-400">Draw</span>
              <span className="tabular-nums w-16 text-right text-neutral-400">{pct(pD)}</span>
              <div className="flex-1 h-3 bg-neutral-800 rounded overflow-hidden">
                <div className="h-full bg-neutral-500" style={{ width: `${pD * 100}%` }} />
              </div>
            </div>
            <div className="flex items-center gap-3">
              <span className="w-44 truncate font-medium">{flag(b)} {b} wins</span>
              <span className="tabular-nums w-16 text-right text-rose-400">{pct(pBW)}</span>
              <div className="flex-1 h-3 bg-neutral-800 rounded overflow-hidden">
                <div className="h-full bg-rose-500" style={{ width: `${pBW * 100}%` }} />
              </div>
            </div>
          </div>
        )}
      </div>
    </section>
  )
}

export default function App() {
  const [data, setData] = useState(null)
  const [err, setErr] = useState(null)

  useEffect(() => {
    fetch('/predictions.json')
      .then((r) => r.json())
      .then(setData)
      .catch((e) => setErr(e.message))
  }, [])

  if (err) {
    return <div className="p-8 text-rose-400">Failed to load predictions: {err}</div>
  }
  if (!data) {
    return <div className="p-8 text-neutral-500">Loading forecast…</div>
  }

  return (
    <div className="min-h-screen">
      <header className="border-b border-neutral-800 bg-neutral-950/95 backdrop-blur sticky top-0 z-10">
        <div className="max-w-6xl mx-auto px-4 py-4 flex items-baseline justify-between flex-wrap gap-2">
          <div>
            <h1 className="text-2xl font-semibold tracking-tight">WC 2026 Forecast</h1>
            <p className="text-xs text-neutral-500 mt-1">{data.model}</p>
          </div>
          <p className="text-xs text-neutral-500">
            {data.n_sims.toLocaleString()} simulations · generated {data.generated_at}
          </p>
        </div>
      </header>

      <main className="max-w-6xl mx-auto px-4 py-6 space-y-10">
        <TopContenders teams={data.team_summary} />
        <GroupGrid groups={data.groups} summary={data.team_summary} />
        <MatchupPicker teams={data.team_summary} matchups={data.matchups} />
      </main>

      <footer className="border-t border-neutral-800 mt-12">
        <div className="max-w-6xl mx-auto px-4 py-6 text-xs text-neutral-500">
          Reg-Dixon-Coles (group stage) + MLP+XGB+CatBoost ensemble (knockouts). Backtest WC 2010-2022:
          log_loss 0.99, accuracy ~55%. Predictions are model outputs, not bookmaker odds.
        </div>
      </footer>
    </div>
  )
}
