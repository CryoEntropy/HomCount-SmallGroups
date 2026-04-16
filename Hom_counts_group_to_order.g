# 辅助函数：CSV 转义（不变）
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

# 同态计数函数（不变）
SimpleHomCount := function(G, H)
    local all_homs, invG, invH, count, i, j;
    
    if not IsFinite(H) then
        return "infinite";
    fi;
    
    if IsFinite(G) and IsAbelian(G) and IsAbelian(H) then
        invG := AbelianInvariants(G);
        invH := AbelianInvariants(H);
        count := 1;
        for i in invG do
            for j in invH do
                count := count * Gcd(i, j);
            od;
        od;
        return count;
    else
        all_homs := AllHomomorphisms(G, H);
        return Length(all_homs);
    fi;
end;

# 修正后的函数：固定源群 → 所有目标阶群
HomCountFromFixedGroupToAllOfOrder := function(G, targetOrder, arg...)
  local numH, Hlist, idsH, descH, homCounts, outfile, fp, parts, i,
        srcId, srcDesc;   # 所有局部变量合并在一起

  # 解析可选参数 outfile
  if Length(arg) = 0 then
    outfile := true;
  else
    outfile := arg[1];
  fi;

  # 检查目标阶
  numH := NumberSmallGroups(targetOrder);
  if numH = fail then
    Print("HomCountFromFixedGroupToAllOfOrder: SmallGroups database does not contain groups of order ", targetOrder, ".\n");
    return;
  fi;

  # 获取源群信息
  if HasIdGroup(G) then
    srcId := IdGroup(G);
    srcDesc := StructureDescription(G);
  else
    srcId := "unknown";
    srcDesc := "unknown";
  fi;

  Print("Source group: ", srcId, " ", srcDesc, "\n");
  Print("Target order: ", targetOrder, " (", numH, " groups)\n");
  Print("Computing homomorphisms...\n\n");

  # 枚举所有目标群
  Hlist := List([1..numH], k -> SmallGroup(targetOrder, k));
  idsH  := List(Hlist, H -> IdGroup(H));
  descH := List(Hlist, H -> StructureDescription(H));

  # 计算同态数量
  homCounts := [];
  for i in [1..numH] do
    homCounts[i] := SimpleHomCount(G, Hlist[i]);
    Print("|Hom(", srcId, " ", srcDesc, " -> ",
          idsH[i], " ", descH[i], ")| = ", homCounts[i], "\n");
  od;

  # 输出 CSV 文件
  if outfile = true then
    outfile := Concatenation("Hom_from_", String(srcId[1]), "_", String(srcId[2]),
                             "_to_order_", String(targetOrder), ".csv");
  fi;

  if outfile <> false then
    fp := OutputTextFile(outfile, false);
    SetPrintFormattingStatus(fp, false);

    # 写入表头
    parts := [EscapeCSV(Concatenation("Source: ", String(srcId), " ", srcDesc))];
    for i in [1..numH] do
      Add(parts, EscapeCSV(Concatenation( String(idsH[i][1]), ".", String(idsH[i][2]), " ", descH[i] )));
    od;
    WriteLine(fp, JoinStringsWithSeparator(parts, ","));

    # 写入数据行
    parts := [EscapeCSV(Concatenation(String(srcId[1]), ".", String(srcId[2]), " ", srcDesc))];
    for i in [1..numH] do
      Add(parts, String(homCounts[i]));
    od;
    WriteLine(fp, JoinStringsWithSeparator(parts, ","));

    CloseStream(fp);
    Print("\nResults saved to ", outfile, "\n");
  fi;
end;