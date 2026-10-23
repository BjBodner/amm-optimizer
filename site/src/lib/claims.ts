import data from '../../data/claims.json';

export type ClaimStatus = 'final' | 'dev' | 'pending';

export interface ClaimRecord {
	label: string;
	value: string;
	status: ClaimStatus;
	requiresFinal: boolean;
	source: string;
	finalSlot?: string;
}

const claims = data.claims as Record<string, ClaimRecord>;

/** The record of a claim; throws on an unknown id so a typo cannot ship. */
export function getClaim(id: string): ClaimRecord {
	const record = claims[id];
	if (!record) throw new Error(`Unknown claim id "${id}" (see data/claims.json)`);
	return record;
}

export function claimValue(id: string): string {
	return getClaim(id).value;
}
