import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "../lib/api";

export default function Politicians() {
  const qc = useQueryClient();
  const politicians = useQuery({ queryKey: ["politicians"], queryFn: () => api.politicians() });

  const follow = useMutation({
    mutationFn: ({ id, followed }: { id: number; followed: boolean }) => api.setFollow(id, followed),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["politicians"] }),
  });

  return (
    <div>
      <h1 className="page-title">Politicians</h1>
      <p className="page-sub">
        Follow a politician to have their newly-disclosed trades generate copy signals.
      </p>

      <div className="card">
        <table>
          <thead>
            <tr><th>Name</th><th>Chamber</th><th>Party</th><th>State</th><th>Follow</th></tr>
          </thead>
          <tbody>
            {(politicians.data ?? []).map((p) => (
              <tr key={p.id}>
                <td><strong>{p.full_name}</strong></td>
                <td><span className={`badge ${p.chamber}`}>{p.chamber}</span></td>
                <td>{p.party ?? "—"}</td>
                <td>{p.state ?? "—"}</td>
                <td>
                  <span
                    className={`toggle ${p.followed ? "on" : ""}`}
                    onClick={() => follow.mutate({ id: p.id, followed: !p.followed })}
                  >
                    {p.followed ? "✓ Following" : "Follow"}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
