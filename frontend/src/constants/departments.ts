export interface DepartmentOption {
  id: string;
  code: string;
  name_en: string;
  name_hi: string;
  short_label: string;
}

export const MASTER_DEPARTMENTS: DepartmentOption[] = [
  {
    id: "forest",
    code: "Forest",
    name_en: "Forest Department",
    name_hi: "वन एवं पर्यावरण",
    short_label: "वन एवं पर्यावरण (Forest)",
  },
  {
    id: "personnel",
    code: "Personnel",
    name_en: "Personnel & General Admin Department",
    name_hi: "कार्मिक एवं सामान्य प्रशासन",
    short_label: "कार्मिक एवं सामान्य प्रशासन (Personnel)",
  },
  {
    id: "finance",
    code: "Finance",
    name_en: "Finance Department",
    name_hi: "वित्त विभाग",
    short_label: "वित्त विभाग (Finance)",
  },
  {
    id: "revenue",
    code: "Revenue",
    name_en: "Revenue Department",
    name_hi: "राजस्व विभाग",
    short_label: "राजस्व विभाग (Revenue)",
  },
  {
    id: "education",
    code: "Education",
    name_en: "Education Department",
    name_hi: "विद्यालयी शिक्षा",
    short_label: "विद्यालयी शिक्षा (Education)",
  },
  {
    id: "rural_dev",
    code: "Rural Development",
    name_en: "Rural Development Department",
    name_hi: "ग्राम्य विकास",
    short_label: "ग्राम्य विकास (Rural Development)",
  },
  {
    id: "urban_dev",
    code: "Urban Development",
    name_en: "Urban Development Department",
    name_hi: "नगर विकास",
    short_label: "नगर विकास (Urban Development)",
  },
  {
    id: "health",
    code: "Health",
    name_en: "Health & Family Welfare Department",
    name_hi: "चिकित्सा एवं स्वास्थ्य",
    short_label: "चिकित्सा एवं स्वास्थ्य (Health)",
  },
  {
    id: "irrigation",
    code: "Irrigation",
    name_en: "Irrigation Department",
    name_hi: "सिंचाई विभाग",
    short_label: "सिंचाई विभाग (Irrigation)",
  },
  {
    id: "transport",
    code: "Transport",
    name_en: "Transport Department",
    name_hi: "परिवहन विभाग",
    short_label: "परिवहन विभाग (Transport)",
  },
  {
    id: "home",
    code: "Home",
    name_en: "Home & Police Department",
    name_hi: "गृह विभाग",
    short_label: "गृह विभाग (Home)",
  },
];
