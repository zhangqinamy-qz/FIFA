import { useEffect, useMemo, useState } from 'react'

// FIFA 3-letter country codes for compact display (bracket boxes etc.).
const SHORT_NAME = {
  Mexico: 'MEX', 'South Africa': 'RSA', 'South Korea': 'KOR', 'Czech Republic': 'CZE',
  Canada: 'CAN', 'Bosnia and Herzegovina': 'BIH', Qatar: 'QAT', Switzerland: 'SUI',
  Brazil: 'BRA', Morocco: 'MAR', Haiti: 'HAI', Scotland: 'SCO',
  'United States': 'USA', Paraguay: 'PAR', Australia: 'AUS', Turkey: 'TUR',
  Germany: 'GER', Curaçao: 'CUW', 'Ivory Coast': 'CIV', Ecuador: 'ECU',
  Netherlands: 'NED', Japan: 'JPN', Sweden: 'SWE', Tunisia: 'TUN',
  Belgium: 'BEL', Egypt: 'EGY', Iran: 'IRN', 'New Zealand': 'NZL',
  Spain: 'ESP', 'Cape Verde': 'CPV', 'Saudi Arabia': 'KSA', Uruguay: 'URU',
  France: 'FRA', Senegal: 'SEN', Iraq: 'IRQ', Norway: 'NOR',
  Argentina: 'ARG', Algeria: 'ALG', Austria: 'AUT', Jordan: 'JOR',
  Portugal: 'POR', 'DR Congo': 'COD', Uzbekistan: 'UZB', Colombia: 'COL',
  England: 'ENG', Croatia: 'CRO', Ghana: 'GHA', Panama: 'PAN',
}
const shortName = (t) => SHORT_NAME[t] || t

// ISO 3166-1 alpha-2 codes (lowercase for flagcdn.com). Subnationals — England
// and Scotland — use flagcdn's special "gb-eng" / "gb-sct" identifiers.
const COUNTRY_CODE = {
  Mexico: 'mx', 'South Africa': 'za', 'South Korea': 'kr', 'Czech Republic': 'cz',
  Canada: 'ca', 'Bosnia and Herzegovina': 'ba', Qatar: 'qa', Switzerland: 'ch',
  Brazil: 'br', Morocco: 'ma', Haiti: 'ht', Scotland: 'gb-sct',
  'United States': 'us', Paraguay: 'py', Australia: 'au', Turkey: 'tr',
  Germany: 'de', Curaçao: 'cw', 'Ivory Coast': 'ci', Ecuador: 'ec',
  Netherlands: 'nl', Japan: 'jp', Sweden: 'se', Tunisia: 'tn',
  Belgium: 'be', Egypt: 'eg', Iran: 'ir', 'New Zealand': 'nz',
  Spain: 'es', 'Cape Verde': 'cv', 'Saudi Arabia': 'sa', Uruguay: 'uy',
  France: 'fr', Senegal: 'sn', Iraq: 'iq', Norway: 'no',
  Argentina: 'ar', Algeria: 'dz', Austria: 'at', Jordan: 'jo',
  Portugal: 'pt', 'DR Congo': 'cd', Uzbekistan: 'uz', Colombia: 'co',
  England: 'gb-eng', Croatia: 'hr', Ghana: 'gh', Panama: 'pa',
}

function Flag({ team, size = 20 }) {
  const code = COUNTRY_CODE[team]
  if (!code) return <span className="text-xs">⚽</span>
  // flagcdn supports w20, w40, w80, w160 ... raster widths
  const widthBucket = size <= 20 ? 'w20' : size <= 40 ? 'w40' : size <= 80 ? 'w80' : 'w160'
  return (
    <img
      src={`https://flagcdn.com/${widthBucket}/${code}.png`}
      srcSet={`https://flagcdn.com/${widthBucket === 'w20' ? 'w40' : 'w80'}/${code}.png 2x`}
      width={size}
      height={Math.round(size * 0.75)}
      alt={team}
      loading="lazy"
      className="inline-block rounded-sm align-middle shadow-sm"
    />
  )
}

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
                  <span className="mr-2 inline-flex"><Flag team={t.team} size={20} /></span>{t.team}
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
            <Flag team={t.team} size={20} />
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
              <option key={t} value={t}>{t}</option>
            ))}
          </select>
          <select
            value={b}
            onChange={(e) => setB(e.target.value)}
            className="bg-neutral-950 border border-neutral-700 rounded px-3 py-2 text-sm"
          >
            {teamNames.map((t) => (
              <option key={t} value={t}>{t}</option>
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
              <span className="w-44 truncate font-medium flex items-center gap-2"><Flag team={a} size={20} /> {a} wins</span>
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
              <span className="w-44 truncate font-medium flex items-center gap-2"><Flag team={b} size={20} /> {b} wins</span>
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

// FIFA 2026 official R32 slot label table — each R32 match has one or two
// "best-third" slots whose pool depends on which 8 of 12 groups produced
// qualifying thirds. The label shown in the bracket box is the canonical pool.
const SLOT_LABEL_BY_MATCH = {
  73: ['2A', '2B'],
  74: ['1E', '3ABCDF'],
  75: ['1F', '2C'],
  76: ['1C', '2F'],
  77: ['1I', '3CDFGH'],
  78: ['2E', '2I'],
  79: ['1A', '3CEFHI'],
  80: ['1L', '3EHIJK'],
  81: ['1D', '3BEFIJ'],
  82: ['1G', '3AEHIJ'],
  83: ['2K', '2L'],
  84: ['1H', '2J'],
  85: ['1B', '3EFGIJ'],
  86: ['1J', '2H'],
  87: ['1K', '3DEIJL'],
  88: ['2D', '2G'],
}

// Which R32 matches sit on the LEFT half of the bracket (per the official
// FIFA wiring: M101 = W97 vs W98 = left semifinal). Right half = the rest.
const LEFT_R32  = [74, 77, 73, 75, 83, 84, 81, 82]
const RIGHT_R32 = [76, 78, 79, 80, 86, 88, 85, 87]
const LEFT_R16  = [89, 90, 93, 94]
const RIGHT_R16 = [91, 92, 95, 96]
const LEFT_QF   = [97, 98]
const RIGHT_QF  = [99, 100]
const LEFT_SF   = 101
const RIGHT_SF  = 102

// Group colors used in the side rails
const GROUP_COLORS = {
  A: 'bg-emerald-600/90', B: 'bg-red-700/90',   C: 'bg-orange-500/90',
  D: 'bg-blue-600/90',    E: 'bg-purple-600/90', F: 'bg-lime-600/90',
  G: 'bg-rose-600/90',    H: 'bg-teal-600/90',  I: 'bg-violet-600/90',
  J: 'bg-amber-600/90',   K: 'bg-orange-600/90', L: 'bg-sky-600/90',
}

function GroupTile({ letter, teams }) {
  return (
    <div className="flex flex-col items-stretch">
      <div className="grid grid-cols-2 gap-px p-0.5 bg-neutral-800 rounded-t border border-neutral-700">
        {teams.map((t) => (
          <div key={t} className="w-8 h-6 bg-neutral-900 flex items-center justify-center" title={t}>
            <Flag team={t} size={22} />
          </div>
        ))}
      </div>
      <div className={`${GROUP_COLORS[letter]} text-white text-[9px] font-semibold uppercase tracking-wider text-center py-0.5 rounded-b`}>
        Group {letter}
      </div>
    </div>
  )
}

function R32Box({ match, side }) {
  if (!match) return <div className="h-14 w-24" />
  const labels = SLOT_LABEL_BY_MATCH[match.match_no]
  const winnerIsA = match.mode_w === match.mode_a
  return (
    <div className="bg-neutral-900 border border-neutral-700 rounded text-[10px] flex flex-col w-24 overflow-hidden shadow">
      <SlotLine label={labels[0]} team={match.mode_a} highlight={winnerIsA} />
      <div className="border-t border-neutral-800" />
      <SlotLine label={labels[1]} team={match.mode_b} highlight={!winnerIsA && match.mode_w === match.mode_b} />
    </div>
  )
}

function SlotLine({ label, team, highlight }) {
  return (
    <div className={`flex items-center gap-1 px-1.5 py-1 ${highlight ? 'bg-neutral-800/80' : ''}`}>
      <span className="text-neutral-500 font-mono text-[9px] w-10 shrink-0 truncate">{label}</span>
      <Flag team={team} size={12} />
      <span className={`font-semibold text-[10px] tabular-nums ${highlight ? 'text-neutral-100' : 'text-neutral-400'}`}>
        {team ? shortName(team) : '—'}
      </span>
    </div>
  )
}

function ResultBox({ match, label, size = 'md' }) {
  if (!match) return null
  const w = match.mode_w
  const p = match.p_mode_w
  const wh = size === 'lg' ? 'w-28 h-20' : size === 'sm' ? 'w-20 h-14' : 'w-24 h-16'
  return (
    <div className={`bg-neutral-900 border border-neutral-700 rounded flex flex-col items-center justify-center gap-0.5 shadow ${wh}`}>
      <div className="text-[8px] uppercase tracking-wider text-neutral-500">{label}</div>
      {w ? (
        <>
          <Flag team={w} size={size === 'lg' ? 24 : 18} />
          <div className={`font-semibold ${size === 'lg' ? 'text-xs' : 'text-[10px]'}`}>{shortName(w)}</div>
          <div className="text-[9px] text-emerald-400 tabular-nums">{pct(p)}</div>
        </>
      ) : <div className="text-[10px] text-neutral-600">TBD</div>}
    </div>
  )
}

function BracketView({ bracket, groups }) {
  if (!bracket) return <p className="text-neutral-500 text-sm">No bracket data.</p>
  const byNo = Object.fromEntries(bracket.matches.map((m) => [m.match_no, m]))
  const groupsLeft  = ['A','B','C','D','E','F']
  const groupsRight = ['G','H','I','J','K','L']
  const final = byNo[104]

  return (
    <section>
      <h2 className="text-lg font-semibold text-neutral-300 mb-1 tracking-wide uppercase text-center">
        Most likely road to the trophy
      </h2>
      <p className="text-xs text-neutral-500 mb-6 text-center">
        Modal team in each slot from {bracket.n_sims.toLocaleString()} sims. Highlighted line = predicted winner of that match.
      </p>

      <div className="pb-6">
        <div className="flex items-center justify-center gap-1.5">

          {/* Left groups */}
          <div className="flex flex-col gap-1.5">
            {groupsLeft.map((g) => <GroupTile key={g} letter={g} teams={groups[g]} />)}
          </div>

          {/* Left R32 */}
          <div className="flex flex-col gap-1">
            <div className="text-[9px] uppercase text-neutral-500 text-center">R32 · M73-84</div>
            {LEFT_R32.map((m) => <R32Box key={m} match={byNo[m]} side="left" />)}
          </div>

          {/* Left R16 */}
          <div className="flex flex-col justify-around" style={{ height: '720px' }}>
            <div className="text-[9px] uppercase text-neutral-500 text-center mb-1">R16 · M89-94</div>
            {LEFT_R16.map((m) => <ResultBox key={m} match={byNo[m]} label={`M${m}`} size="sm" />)}
          </div>

          {/* Left QF */}
          <div className="flex flex-col justify-around" style={{ height: '720px' }}>
            <div className="text-[9px] uppercase text-neutral-500 text-center mb-1">QF · M97-98</div>
            {LEFT_QF.map((m) => <ResultBox key={m} match={byNo[m]} label={`M${m}`} size="md" />)}
          </div>

          {/* Left SF */}
          <div className="flex flex-col justify-center" style={{ height: '720px' }}>
            <div className="text-[9px] uppercase text-neutral-500 text-center mb-1">SF · M101</div>
            <ResultBox match={byNo[LEFT_SF]} label="M101" size="md" />
          </div>

          {/* CENTER: trophy / final / champion */}
          <div className="flex flex-col items-center justify-center gap-2 px-4">
            <div className="text-[10px] uppercase tracking-widest text-amber-400">World Champion</div>
            <div className="w-36 h-36 rounded-lg border border-amber-500/40 bg-gradient-to-b from-amber-900/30 to-amber-700/10 flex flex-col items-center justify-center gap-1">
              <div className="text-3xl">🏆</div>
              {final?.mode_w ? (
                <>
                  <Flag team={final.mode_w} size={36} />
                  <div className="text-sm font-semibold">{final.mode_w}</div>
                  <div className="text-xs text-amber-400 tabular-nums">{pct(final.p_mode_w)}</div>
                </>
              ) : <div className="text-xs text-neutral-500">TBD</div>}
            </div>
            <div className="text-[9px] uppercase tracking-widest text-neutral-500 mt-2">Final · M104</div>
            <div className="text-[10px] text-neutral-400 text-center max-w-[180px]">
              Most likely final:<br />
              <span className="font-medium text-neutral-200">{final?.mode_a}</span>
              <span className="text-neutral-500 mx-1">vs</span>
              <span className="font-medium text-neutral-200">{final?.mode_b}</span>
            </div>
          </div>

          {/* Right SF */}
          <div className="flex flex-col justify-center" style={{ height: '720px' }}>
            <div className="text-[9px] uppercase text-neutral-500 text-center mb-1">SF · M102</div>
            <ResultBox match={byNo[RIGHT_SF]} label="M102" size="md" />
          </div>

          {/* Right QF */}
          <div className="flex flex-col justify-around" style={{ height: '720px' }}>
            <div className="text-[9px] uppercase text-neutral-500 text-center mb-1">QF · M99-100</div>
            {RIGHT_QF.map((m) => <ResultBox key={m} match={byNo[m]} label={`M${m}`} size="md" />)}
          </div>

          {/* Right R16 */}
          <div className="flex flex-col justify-around" style={{ height: '720px' }}>
            <div className="text-[9px] uppercase text-neutral-500 text-center mb-1">R16 · M91-96</div>
            {RIGHT_R16.map((m) => <ResultBox key={m} match={byNo[m]} label={`M${m}`} size="sm" />)}
          </div>

          {/* Right R32 */}
          <div className="flex flex-col gap-1">
            <div className="text-[9px] uppercase text-neutral-500 text-center">R32 · M76-88</div>
            {RIGHT_R32.map((m) => <R32Box key={m} match={byNo[m]} side="right" />)}
          </div>

          {/* Right groups */}
          <div className="flex flex-col gap-1.5">
            {groupsRight.map((g) => <GroupTile key={g} letter={g} teams={groups[g]} />)}
          </div>

        </div>
      </div>

      <p className="text-[10px] text-neutral-500 text-center mt-4">
        Slot codes follow the official FIFA 2026 template (e.g. 3ABCDF = best-third from one of groups A, B, C, D, or F).
        Team in each slot is the modal occupant from {bracket.n_sims.toLocaleString()} simulations.
      </p>
    </section>
  )
}

const TABS = [
  { id: 'overview', label: 'Overview' },
  { id: 'bracket',  label: 'Bracket' },
]

export default function App() {
  const [data, setData] = useState(null)
  const [err, setErr] = useState(null)
  const [tab, setTab] = useState('overview')

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
        <nav className="max-w-6xl mx-auto px-4 flex gap-1 border-t border-neutral-800/60">
          {TABS.map((t) => (
            <button
              key={t.id}
              onClick={() => setTab(t.id)}
              className={`px-4 py-2 text-sm border-b-2 -mb-px transition-colors ${
                tab === t.id
                  ? 'border-emerald-500 text-emerald-400'
                  : 'border-transparent text-neutral-500 hover:text-neutral-300'
              }`}
            >
              {t.label}
            </button>
          ))}
        </nav>
      </header>

      <main className={`${tab === 'bracket' ? 'max-w-screen-2xl' : 'max-w-6xl'} mx-auto px-4 py-6 space-y-10`}>
        {tab === 'overview' && (
          <>
            <TopContenders teams={data.team_summary} />
            <GroupGrid groups={data.groups} summary={data.team_summary} />
            <MatchupPicker teams={data.team_summary} matchups={data.matchups} />
          </>
        )}
        {tab === 'bracket' && <BracketView bracket={data.bracket} groups={data.groups} />}
      </main>

      <footer className="border-t border-neutral-800 mt-12">
        <div className="max-w-6xl mx-auto px-4 py-6 text-xs text-neutral-500">
          Reg-Dixon-Coles (group stage) + MLP+XGB+CatBoost ensemble (knockouts). Backtest WC 2010-2022:
          log_loss 0.99, accuracy ~55%. Predictions are model outputs, not bookmaker odds. Match numbers follow the
          official FIFA 2026 schedule (M73-M104 = knockout stage).
        </div>
      </footer>
    </div>
  )
}
