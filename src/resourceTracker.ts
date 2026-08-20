export type OIDNResourceKind =
  | 'gpu-buffer'
  | 'gpu-query-set'
  | 'ml-context'
  | 'ml-graph'
  | 'ml-tensor';

export interface OIDNResourceStats {
  created: number;
  destroyed: number;
  live: number;
  peakLive: number;
}

export interface OIDNResourceSnapshot {
  live: number;
  created: number;
  destroyed: number;
  peakLive: number;
  pending: number;
  byKind: Partial<Record<OIDNResourceKind, OIDNResourceStats>>;
}

interface MutableResourceStats extends OIDNResourceStats {
  resources: Set<object>;
}

/**
 * Counts explicitly owned browser resources. Keeping the ownership registry in
 * production makes cache eviction and failure cleanup testable without relying
 * on JS heap measurements, which do not include WebGPU allocations.
 */
export class OIDNResourceTracker {
  private _stats = new Map<OIDNResourceKind, MutableResourceStats>();

  track<T extends object>(kind: OIDNResourceKind, resource: T): T {
    let stats = this._stats.get(kind);
    if (!stats) {
      stats = {
        created: 0,
        destroyed: 0,
        live: 0,
        peakLive: 0,
        resources: new Set()
      };
      this._stats.set(kind, stats);
    }
    if (stats.resources.has(resource)) return resource;
    stats.resources.add(resource);
    stats.created++;
    stats.live++;
    stats.peakLive = Math.max(stats.peakLive, stats.live);
    return resource;
  }

  release(
    kind: OIDNResourceKind,
    resource: object | undefined,
    destroy: () => void
  ) {
    if (!resource) return false;
    const stats = this._stats.get(kind);
    if (!stats?.resources.delete(resource)) return false;
    try {
      destroy();
    } finally {
      stats.destroyed++;
      stats.live--;
    }
    return true;
  }

  snapshot(pending = 0): OIDNResourceSnapshot {
    let live = 0;
    let created = 0;
    let destroyed = 0;
    let peakLive = 0;
    const byKind: OIDNResourceSnapshot['byKind'] = {};
    for (const [kind, stats] of this._stats) {
      const entry = {
        created: stats.created,
        destroyed: stats.destroyed,
        live: stats.live,
        peakLive: stats.peakLive
      };
      byKind[kind] = entry;
      live += entry.live;
      created += entry.created;
      destroyed += entry.destroyed;
      peakLive += entry.peakLive;
    }
    return { live, created, destroyed, peakLive, pending, byKind };
  }
}
