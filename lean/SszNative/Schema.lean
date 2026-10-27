import SszNative.Branch

/-!
Declarations compiled once, so that each value read against them skips walking the declaration again.

A struct of fixed-size fields compiles to its width, its probes, and where each field's leaf comes from.
Its root is then a handful of copies and one hashing call per tree level.
Every other declaration keeps the general path.
-/

namespace Ssz.Native

open Ssz

/-- Where one leaf of a fixed-size struct comes from. -/
inductive Leaf where
  /-- A field that is its own root: its bytes, zero-padded to one node. -/
  | bytes (offset width : Nat)
  /-- A field with a tree of its own, rooted from its bytes. -/
  | nested (d : Desc) (offset width : Nat)

/--
The internal cache produced by `Schema.compile` for a fixed-size struct.

Its public fields do not establish that they describe any particular declaration.
Use the compiler-produced cache for the schema equivalence guarantees.
-/
structure FixedStruct where
  /-- The encoding's width in bytes. -/
  width : Nat
  /-- Whether deserializing it first bounds its size by what four-byte offsets can name. -/
  composite : Bool
  /-- The bytes a value can be refused for, in the order deserialization reads them. -/
  probes : List Probe
  /-- Each field's leaf, in order. -/
  leaves : Array Leaf
  /-- The height of the tree over the leaves. -/
  depth : Nat

/--
A declaration with a cached plan for reading its encodings.

The equivalence theorems concern `Schema.compile d`, not arbitrary public record values.
Construct with `Schema.compile` rather than supplying cache fields independently.
Root operations additionally require a well-formed declaration and the documented size bound.
-/
structure Schema where
  /-- The declaration. -/
  desc : Desc
  /-- The compiled struct, when the declaration is a struct of fixed-size fields. -/
  fixed : Option FixedStruct

/-- The leaves of a struct's fields, from an offset on. -/
def leavesOf : List Desc → Nat → List Leaf
  | [], _ => []
  | field :: rest, offset =>
    let leaf := if packsIntoNode field then Leaf.bytes offset (fixedWidth field)
      else Leaf.nested field offset (fixedWidth field)
    leaf :: leavesOf rest (offset + fixedWidth field)

/--
Builds a cache consistent with a declaration, without checking declaration legality.

Check `d.wellFormed` once before using an externally supplied descriptor for roots or proofs.
Encoding validation still agrees with deserialization for every descriptor.
-/
def Schema.compile (d : Desc) : Schema :=
  match d with
  | .container _ fields =>
    if plain d then
      let compiled : FixedStruct :=
        { width := fixedWidth d, composite := composite d, probes := plan d,
          leaves := (leavesOf fields 0).toArray, depth := depthFor fields.length }
      { desc := d, fixed := some compiled }
    else { desc := d, fixed := none }
  | _ => { desc := d, fixed := none }

/-- Zero nodes enough for any struct's leaves, copied rather than built for each value. -/
def zeroLeaves : ByteArray :=
  zeros (32 * 65)

/-- One record's leaves, written into a buffer of zero nodes. -/
def fillLeaves (b : ByteArray) (leaves : Array Leaf) : ByteArray :=
  -- Room for every leaf and one more node, which an odd level needs.
  let size := 32 * leaves.size + 32
  let buffer := if size ≤ zeroLeaves.size then zeroLeaves.extract 0 size else zeros size
  (leaves.foldl (fun (acc : ByteArray × Nat) leaf =>
    let (buffer, node) := acc
    match leaf with
    | .bytes offset width => (b.copySlice offset buffer (32 * node) width, node + 1)
    | .nested d offset width => ((rootsOf d b offset width 1).copySlice 0 buffer (32 * node) 32, node + 1))
    (buffer, 0)).1

/--
A short level of nodes folded the remaining levels up, from a given height.

The level is owned and has room for one more node.
An odd level therefore takes its zero tree in place, and each level is one hashing call.
-/
def foldShort (level : ByteArray) (width height : Nat) : (remaining : Nat) → ByteArray
  | 0 => if level.size == 32 then level else level.extract 0 32
  | remaining + 1 =>
    if width ≤ 1 then climb (level.extract 0 32) height (remaining + 1)
    else
      let padded := if width % 2 == 1 then (zeroRoot height).copySlice 0 level (32 * width) 32 else level
      foldShort (hashPairs padded 0 ((width + 1) / 2).toUSize) ((width + 1) / 2) (height + 1) remaining

/--
For `s = Schema.compile d`, refuses exactly the encodings deserialization refuses.

This checks encoding bytes; it does not establish declaration legality or validate forged caches.
-/
def Schema.validate (s : Schema) (b : ByteArray) : Except Err Unit :=
  match s.fixed with
  | some f =>
    let size := b.size
    if f.composite && size ≥ 4294967296 then .error (.offsetOverflow size)
    else if size != f.width then .error (.scope f.width size)
    -- A passing value needs no refusal built.
    else if probesPass b 0 f.probes then .ok ()
    else checkProbes b 0 f.probes
  | none => Native.validate s.desc b

/--
Unchecked rooting of an already accepted encoding with a compiler-produced schema.

Requires a well-formed declaration, successful encoding validation and
`64 * b.size + 128 < USize.size` for reference equivalence.
-/
def Schema.rootOf (s : Schema) (b : ByteArray) : ByteArray :=
  match s.fixed with
  | some f => foldShort (fillLeaves b f.leaves) f.leaves.size 0 f.depth
  | none => Native.rootOf s.desc b 0 b.size

/--
Roots an encoding with a compiler-produced schema, or reports its encoding refusal.

For reference equivalence, the declaration must be well formed and
`64 * b.size + 128 < USize.size`. Encoding validation does not check these premises.
-/
def Schema.hashTreeRoot (s : Schema) (b : ByteArray) : Except Err ByteArray := do
  s.validate b
  return s.rootOf b

end Ssz.Native
