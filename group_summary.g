# 辅助函数：CSV 转义（来自 Hom_counts_group_to_order.g）
EscapeCSV := function(s)
  local str;
  str := String(s);
  if ForAny(str, c -> c in ",\"\n\r") then
    str := ReplacedString(str, "\"", "\"\"");
    return Concatenation("\"", str, "\"");
  else
    return str;
  fi;
end;

list := function(order, arg...)
  local m, G, i, id, desc, expo, centOrd, derOrd, abInv, abRank, abelOrd, nilpClass,
        fp, parts, outfile, ids, descs, expos, centOrds, derOrds, abInvs, abRanks, abelOrds, nilpClasses;;

  # 解析可选参数 outfile
  if Length(arg) = 0 then
    outfile := true;
  else
    outfile := arg[1];
  fi;

  m := NumberSmallGroups(order);
  if m = fail then
    Print("list: SmallGroups database does not contain groups of order ", order, ".\n");
    return;
  fi;

  Print("There are ", m, " groups of order ", order, ".\n\n");

  # 准备收集数据
  ids := [];
  descs := [];
  expos := [];
  centOrds := [];
  derOrds := [];
  abInvs := [];
  abRanks := [];
  abelOrds := [];
  nilpClasses := [];

  for i in [1..m] do
    G := SmallGroup(order, i);
    id := IdGroup(G);
    desc := StructureDescription(G);
    expo := Exponent(G);
    centOrd := Size(Center(G));
    derOrd := Size(DerivedSubgroup(G));
    abInv := AbelianInvariants(G / DerivedSubgroup(G));
    abRank := Length(abInv);
    abelOrd := Size(G / DerivedSubgroup(G));
    if IsNilpotentGroup(G) then
        nilpClass := NilpotencyClassOfGroup(G);
    else
        nilpClass := "not nilpotent";
    fi;

    # 打印到控制台
    Print("Id: ", id, "\n");
    Print("StructureDescription: ", desc, "\n");
    Print("Exponent: ", expo, ", CenterOrder: ", centOrd,
          ", DerivedOrder: ", derOrd, "\n");
    Print("AbelianInvariants(G/[G,G]): ", abInv, "\n");
    Print("the order of the abelianization: ", abelOrd, "\n");
    Print("NilpotencyClass: ");
    if IsString(nilpClass) then
        Print(nilpClass, "\n");
    else
        Print(nilpClass, "\n");
    fi;
    Print("\n");

    # 存储数据
    Add(ids, id);
    Add(descs, desc);
    Add(expos, expo);
    Add(centOrds, centOrd);
    Add(derOrds, derOrd);
    Add(abInvs, abInv);
    Add(abRanks, abRank);
    Add(abelOrds, abelOrd);
    Add(nilpClasses, nilpClass);
  od;

  # 生成 CSV 文件
  if outfile = true then
    outfile := Concatenation("groups_order_", String(order), "_summary.csv");
  fi;

  if outfile <> false then
    fp := OutputTextFile(outfile, false);
    SetPrintFormattingStatus(fp, false);

    # 表头（与 hom32_group_summary.csv 顺序一致）
    parts := ["Id", "Name", "Exponent", "CenterOrder", "DerivedOrder",
              "AbelianInvariants", "AbelRank", "AbelianizationOrder", "NilpotencyClass"];
    WriteLine(fp, JoinStringsWithSeparator(List(parts, EscapeCSV), ","));

    # 数据行
    for i in [1..m] do
      parts := [
        Concatenation(String(ids[i][1]), ".", String(ids[i][2])),
        EscapeCSV(descs[i]),
        String(expos[i]),
        String(centOrds[i]),
        String(derOrds[i]),
        EscapeCSV(String(abInvs[i])),
        String(abRanks[i]),
        String(abelOrds[i]),
        EscapeCSV(String(nilpClasses[i]))
      ];
      WriteLine(fp, JoinStringsWithSeparator(parts, ","));
    od;

    CloseStream(fp);
    Print("\nResults saved to ", outfile, "\n");
  fi;
end;