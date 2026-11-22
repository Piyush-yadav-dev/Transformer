import test from "node:test";
import assert from "node:assert/strict";
import { analytics_charts } from "../../frontend/application.js";
const {segments,chartDomain} = analytics_charts;
test("raw chart segmentation never bridges missing optimizer updates",()=>{
 const points=[{tokens:8,loss:3},{tokens:16,loss:2},{tokens:32,loss:1,gap_before:true}];
 assert.deepEqual(segments(points),[points.slice(0,2),points.slice(2)]);
 assert.equal(points[0].loss,3);
});
test("chart domain includes every finite raw peak and both runs without smoothing",()=>{
 const series=[{points:[{tokens:10,loss:5},{tokens:20,loss:1}]},{points:[{tokens:30,loss:8}]}];
 assert.deepEqual(chartDomain(series),{x:30,y:8.4});
 assert.deepEqual(chartDomain([]),{x:1,y:1.05});
});
