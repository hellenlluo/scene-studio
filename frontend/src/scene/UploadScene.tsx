import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useRef, useState } from "react";

import { api } from "../api/client";
import { useSceneStore } from "./store";

const TERMINAL_STATES = new Set(["succeeded", "needs_attention", "failed"]);

/** Upload a source photograph, then follow the asynchronous reconstruction job. */
export function UploadScene() {
  const input = useRef<HTMLInputElement>(null);
  const selectedJob = useRef<string | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);
  const queryClient = useQueryClient();
  const setSceneId = useSceneStore((state) => state.setSceneId);

  const upload = useMutation({
    mutationFn: api.createJob,
    onMutate: () => setJobId(null),
    onSuccess: (job) => setJobId(job.id),
  });

  const job = useQuery({
    queryKey: ["job", jobId],
    queryFn: () => api.getJob(jobId as string),
    enabled: jobId !== null,
    refetchInterval: (query) => {
      const state = query.state.data?.state;
      return state && TERMINAL_STATES.has(state) ? false : 1000;
    },
  });

  useEffect(() => {
    const finished = job.data;
    if (!finished?.scene_id || selectedJob.current === finished.id) return;
    selectedJob.current = finished.id;
    // The scene endpoint is already available when the job publishes scene_id.
    // Select it immediately, and refresh the left-hand list so its summary appears.
    setSceneId(finished.scene_id);
    void queryClient.invalidateQueries({ queryKey: ["scenes"] });
  }, [job.data, queryClient, setSceneId]);

  const busy =
    upload.isPending ||
    job.data?.state === "pending" ||
    job.data?.state === "running";
  const stage = job.data?.current_stage?.replaceAll("_", " ");
  const error = upload.error ?? job.error;

  return (
    <div className="scene-upload">
      <input
        ref={input}
        className="scene-upload-input"
        type="file"
        accept="image/jpeg,image/png,image/webp"
        onChange={(event) => {
          const file = event.target.files?.[0];
          event.target.value = "";
          if (file) upload.mutate(file);
        }}
      />
      <button
        type="button"
        className="scene-upload-button"
        disabled={busy}
        onClick={() => input.current?.click()}
      >
        {upload.isPending
          ? "uploading…"
          : busy
            ? `processing${stage ? `: ${stage}` : "…"}`
            : "+ upload photo"}
      </button>
      <p className="note">JPEG, PNG or WebP · maximum 20 MB</p>
      {job.data?.state === "failed" && (
        <p className="note note--error">
          {job.data.error ?? "reconstruction failed"}
        </p>
      )}
      {error && <p className="note note--error">{String(error)}</p>}
    </div>
  );
}
